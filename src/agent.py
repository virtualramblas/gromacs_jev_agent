"""
Milestone 4: SmolAgent & Local SLM Integration.
Wires up the tool library and state manager to a CodeAgent instance
driven by a local SLM (via Ollama) for high-level pipeline orchestration.
"""

import json
from typing import List, Dict, Any, Optional

# Top-level imports compatible with latest smolagents and litellm versions
from smolagents import Tool, CodeAgent, LiteLLMModel

# Import modules from previous milestones
from src.simple_jev import SimpleJevEngine, HardwareConfig
from src.gromacs_tools import GromacsToolLibrary
from src.state_manager import StateManager, MDPGenerator


# =====================================================================
# 1. SmolAgent Tool Definitions (Wrappers around GromacsToolLibrary)
# =====================================================================

class GromacsPipelineTool(Tool):
    """Base tool to provide shared access to the GROMACS tool library and state."""
    def __init__(self, tool_library: GromacsToolLibrary, state_manager: StateManager):
        super().__init__()
        self.tools = tool_library
        self.state = state_manager


class RunPdb2gmxTool(GromacsPipelineTool):
    name = "run_pdb2gmx"
    description = (
        "Generates GROMACS topology (.top) and coordinate (.gro) files from an input PDB file. "
        "This is the first step of the pipeline."
    )
    inputs = {
        "pdb_file": {
            "type": "string",
            "description": "Path to the input protein .pdb file."
        }
    }
    output_type = "string"

    def forward(self, pdb_file: str) -> str:
        result = self.tools.run_pdb2gmx(pdb_file=pdb_file)
        self.state.record_step_result(
            step_name=result.job_id,
            success=result.success,
            output_files=result.created_outputs,
            error_msg=None if result.success else result.stderr_tail
        )
        return json.dumps(result.to_slm_payload())


class RunEditconfTool(GromacsPipelineTool):
    name = "run_editconf"
    description = (
        "Defines the simulation box dimensions and centers the protein coordinates. "
        "Must be run after topology generation."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        gro_file = self.state.get_latest_artifact("pdb2gmx", "gro")
        if not gro_file:
            return json.dumps({
                "status": "failed",
                "error": "Prerequisite 'processed.gro' not found. Run 'run_pdb2gmx' first."
            })
        
        result = self.tools.run_editconf(gro_file=gro_file)
        self.state.record_step_result(
            step_name=result.job_id,
            success=result.success,
            output_files=result.created_outputs,
            error_msg=None if result.success else result.stderr_tail
        )
        return json.dumps(result.to_slm_payload())


class RunSolvateTool(GromacsPipelineTool):
    name = "run_solvate"
    description = (
        "Hydrates the simulation box with solvent (water) molecules. "
        "Must be run after the simulation box has been defined."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        gro_file = self.state.get_latest_artifact("editconf", "gro")
        top_file = self.state.get_latest_artifact("pdb2gmx", "top")
        if not gro_file or not top_file:
            return json.dumps({
                "status": "failed",
                "error": "Prerequisite 'boxed.gro' or 'topol.top' not found. Complete previous steps first."
            })

        result = self.tools.run_solvate(boxed_gro=gro_file, top_file=top_file)
        self.state.record_step_result(
            step_name=result.job_id,
            success=result.success,
            output_files=result.created_outputs,
            error_msg=None if result.success else result.stderr_tail
        )
        return json.dumps(result.to_slm_payload())


class RunMdrunTool(GromacsPipelineTool):
    name = "run_mdrun"
    description = (
        "Executes a molecular dynamics simulation or minimization step (e.g., 'em', 'nvt', 'npt', 'md') "
        "using a pre-compiled .tpr file. This is a computationally intensive step."
    )
    inputs = {
        "step_name": {
            "type": "string",
            "description": "The name of the simulation step, e.g., 'em' (minimization) or 'nvt' (equilibration)."
        }
    }
    output_type = "string"

    def forward(self, step_name: str) -> str:
        # In the complete flow, a grompp step compiles the .tpr from coordinates and templates.
        # For Milestone 4, we retrieve the generated .tpr file for this phase.
        tpr_file = self.state.get_latest_artifact(f"grompp_{step_name}", "tpr")
        if not tpr_file:
            # Fallback path if files are structured flatly under the step name
            tpr_file = self.state.get_latest_artifact(step_name, "tpr")
            
        if not tpr_file:
            return json.dumps({
                "status": "failed",
                "error": f"Prerequisite '{step_name}.tpr' not found. Run the corresponding 'grompp' preparation first."
            })
            
        result = self.tools.run_mdrun(step_name=step_name, tpr_file=tpr_file)
        self.state.record_step_result(
            step_name=result.job_id,
            success=result.success,
            output_files=result.created_outputs,
            error_msg=None if result.success else result.stderr_tail
        )
        return json.dumps(result.to_slm_payload())


# =====================================================================
# 2. Main Agent Orchestrator
# =====================================================================

class GromacsAgent:
    """Orchestrates the entire GROMACS simulation pipeline using an SLM-driven agent."""
    
    def __init__(
        self,
        workdir: str,
        simulation_id: str,
        hardware_config: HardwareConfig,
        model_id: str = "ollama/llama3:8b"
    ):
        # Initialize core engines and managers
        self.jev_engine = SimpleJevEngine(base_workdir=workdir)
        self.tool_library = GromacsToolLibrary(jev_engine=self.jev_engine, hardware_config=hardware_config)
        self.state = StateManager(workdir=workdir, simulation_id=simulation_id)
        self.mdp_gen = MDPGenerator()

        # Define the tools available to the agent's action space
        tools = [
            RunPdb2gmxTool(self.tool_library, self.state),
            RunEditconfTool(self.tool_library, self.state),
            RunSolvateTool(self.tool_library, self.state),
            RunMdrunTool(self.tool_library, self.state),
        ]
        
        # Define and store the System Prompt as a class attribute for robust testing
        self.system_prompt = f"""You are GROMACS-GPT, an expert AI assistant that automates molecular dynamics simulations.
Your goal is to execute the standard GROMACS workflow step-by-step, starting from a PDB file.
The required sequence of steps is: {', '.join(self.state.STAGES_ORDER)}.

INSTRUCTIONS:
1.  Examine the 'CURRENT STATE' to understand what step you are on.
2.  NEVER skip a step. Execute the tools in the correct order.
3.  The `run_mdrun` tool is only for steps like 'em', 'nvt', 'npt', and 'md'.
4.  After each tool call, you will receive a JSON status. If it indicates success, proceed to the next step. If it fails, stop and report the error.
"""

        # Bind the agent to our local SLM via LiteLLMModel
        model = LiteLLMModel(model_id=model_id, system_prompt=self.system_prompt)
        
        # Initialize the smolagents CodeAgent
        self.agent = CodeAgent(tools=tools, model=model, max_steps=15)

    def run_pipeline(self, pdb_file: str):
        """Starts and runs the agentic pipeline from a given input PDB file."""
        print(f"🚀 Starting GROMACS pipeline for {pdb_file}...")
        
        self.state.data["input_pdb"] = pdb_file
        self.state.save()
        
        while self.state.data["status"] not in ["completed", "halted"]:
            # 1. Capture current simulation state
            context = self.state.get_slm_context()
            
            # 2. Build the execution prompt with state tracking
            user_prompt = f"""CURRENT STATE: {json.dumps(context, indent=2)}
            
            Based on the current state, what is the exact next tool I must call? Call it now."""

            print(f"\n--- Agent Turn (Current Step: {context['current_step']}) ---")
            print("Prompting agent with current state...")
            
            # 3. Request action from the local SLM
            try:
                response = self.agent.run(user_prompt)
                print(f"Agent observation: {response}")
            except Exception as e:
                print(f"🚨 Agent execution failed: {e}")
                self.state.record_step_result(
                    step_name=self.state.data['current_step'],
                    success=False,
                    error_msg=str(e)
                )
                break

        print(f"\n✅ Pipeline finished with status: {self.state.data['status']}")
        if self.state.data['status'] == 'halted':
            print(f"🚨 Error details: {self.state.data['errors'][-1]}")