"""
Milestone 4: SmolAgent & Local SLM Integration.
Wires up the tool library and state manager to a `smol-agent` instance
driven by a local SLM (via Ollama) for high-level orchestration.
"""
import json
from typing import List, Dict, Any

# Ensure you have installed smol-agent and litellm:
# pip install -U smol-agent litellm
from smolagents import Tool, CodeAgent
from smolagents.models.litellm import LiteLLMModel

# Import modules from previous milestones
from src.simple_jev import SimpleJevEngine, HardwareConfig
from src.gromacs_tools import GromacsToolLibrary
from src.state_manager import StateManager, MDPGenerator

# =====================================================================
# 1. SmolAgent Tool Definitions (Wrappers around GromacsToolLibrary)
# =====================================================================

class GromacsPipelineTool(Tool):
    """Base tool to provide shared access to the GROMACS tool library."""
    def __init__(self, tool_library: GromacsToolLibrary, state_manager: StateManager):
        super().__init__()
        self.tools = tool_library
        self.state = state_manager

class RunPdb2gmxTool(GromacsPipelineTool):
    name: str = "run_pdb2gmx"
    description: str = "Generates GROMACS topology (.top) and coordinate (.gro) files from a PDB input file. This is the first step."
    inputs: Dict[str, Any] = {
        "pdb_file": {"type": "string", "description": "Path to the input protein .pdb file."}
    }
    output_type: str = "string"

    def forward(self, pdb_file: str) -> str:
        result = self.tools.run_pdb2gmx(pdb_file=pdb_file)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

class RunEditconfTool(GromacsPipelineTool):
    name: str = "run_editconf"
    description: str = "Defines the simulation box, centering the molecule."
    inputs: Dict[str, Any] = {}
    output_type: str = "string"

    def forward(self) -> str:
        gro_file = self.state.get_latest_artifact("pdb2gmx", "gro")
        if not gro_file:
            return json.dumps({"status": "failed", "error": "Prerequisite 'processed.gro' not found. Run `run_pdb2gmx` first."})
        
        result = self.tools.run_editconf(gro_file=gro_file)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

class RunSolvateTool(GromacsPipelineTool):
    name: str = "run_solvate"
    description: str = "Fills the simulation box with water."
    inputs: Dict[str, Any] = {}
    output_type: str = "string"

    def forward(self) -> str:
        gro_file = self.state.get_latest_artifact("editconf", "gro")
        top_file = self.state.get_latest_artifact("pdb2gmx", "top")
        if not gro_file or not top_file:
            return json.dumps({"status": "failed", "error": "Prerequisite 'boxed.gro' or 'topol.top' not found. Complete previous steps first."})

        result = self.tools.run_solvate(boxed_gro=gro_file, top_file=top_file)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())
        
# ... Other tool wrappers for genion, grompp, and mdrun would follow a similar pattern ...
class RunMdrunTool(GromacsPipelineTool):
    name: str = "run_mdrun"
    description: str = "Executes an MD simulation step (e.g., 'em', 'nvt', 'npt', 'md') using a .tpr file. This is a computationally intensive step."
    inputs: Dict[str, Any] = {
        "step_name": {"type": "string", "description": "The name of the simulation step, e.g., 'em', 'nvt'."}
    }
    output_type: str = "string"

    def forward(self, step_name: str) -> str:
        # In a full implementation, the grompp tool would be called first to generate this
        tpr_file = self.state.get_latest_artifact(f"grompp_{step_name}", "tpr")
        if not tpr_file:
            return json.dumps({"status": "failed", "error": f"Prerequisite '{step_name}.tpr' not found. Run the corresponding `grompp` tool first."})
            
        result = self.tools.run_mdrun(step_name=step_name, tpr_file=tpr_file)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

# =====================================================================
# 2. Main Agent Orchestrator
# =====================================================================

class GromacsAgent:
    """Orchestrates the entire GROMACS simulation pipeline using an SLM-driven agent."""
    def __init__(self, workdir: str, simulation_id: str, hardware_config: HardwareConfig, model_id: str = "ollama/llama3:8b"):
        # Initialize all components
        self.jev_engine = SimpleJevEngine(base_workdir=workdir)
        self.tool_library = GromacsToolLibrary(jev_engine=self.jev_engine, hardware_config=hardware_config)
        self.state = StateManager(workdir=workdir, simulation_id=simulation_id)
        self.mdp_gen = MDPGenerator()

        # Define the tools available to the agent
        tools = [
            RunPdb2gmxTool(self.tool_library, self.state),
            RunEditconfTool(self.tool_library, self.state),
            RunSolvateTool(self.tool_library, self.state),
            # Add RunGenionTool, RunGromppTool here...
            RunMdrunTool(self.tool_library, self.state),
        ]
        
        # Define the System Prompt
        system_prompt = f"""You are GROMACS-GPT, an expert AI assistant that automates molecular dynamics simulations.
        Your goal is to execute the standard GROMACS workflow step-by-step, starting from a PDB file.
        The required sequence of steps is: {', '.join(self.state.STAGES_ORDER)}.
        
        INSTRUCTIONS:
        1.  Examine the 'CURRENT STATE' to understand what step you are on.
        2.  NEVER skip a step. Execute the tools in the correct order.
        3.  The `run_mdrun` tool is only for steps like 'em', 'nvt', 'npt', and 'md'.
        4.  After each tool call, you will receive a JSON status. If it indicates success, proceed to the next step. If it fails, stop and report the error.
        """

        # Bind to local SLM via Ollama/LiteLLM
        model = LiteLLMModel(model_id=model_id, system_prompt=system_prompt)
        
        # Initialize the smol-agent
        self.agent = CodeAgent(tools=tools, model=model, max_steps=15)

    def run_pipeline(self, pdb_file: str):
        """Starts and runs the agentic pipeline from a given PDB file."""
        print(f"🚀 Starting GROMACS pipeline for {pdb_file}...")
        
        # Set the initial PDB file in the state
        self.state.data["input_pdb"] = pdb_file
        self.state.save()
        
        while self.state.data["status"] not in ["completed", "halted"]:
            # 1. Get the current state summary for the SLM
            context = self.state.get_slm_context()
            
            # 2. Formulate the prompt for the agent's turn
            user_prompt = f"""CURRENT STATE: {json.dumps(context, indent=2)}
            
            Based on the current state, what is the exact next tool I must call? Call it now."""

            print(f"\n--- Agent Turn (Current Step: {context['current_step']}) ---")
            print(f"Prompting agent with current state...")
            
            # 3. Execute one step of the agent loop
            try:
                response = self.agent.run(user_prompt)
                print(f"Agent observation: {response}")
            except Exception as e:
                print(f"🚨 Agent execution failed: {e}")
                self.state.record_step_result(self.state.data['current_step'], success=False, error_msg=str(e))
                break

        print(f"\n✅ Pipeline finished with status: {self.state.data['status']}")
        if self.state.data['status'] == 'halted':
            print(f"🚨 Error details: {self.state.data['errors'][-1]}")

# =====================================================================
# 3. Example Usage (Test for Milestone 4)
# =====================================================================
if __name__ == "__main__":
    print("--- Verifying Milestone 4: SmolAgent & SLM Integration ---")

    # Ensure you have a test PDB file, e.g., '1AKI.pdb'
    # You can download it from the PDB: `wget https://files.rcsb.org/download/1AKI.pdb`
    TEST_PDB = "1AKI.pdb"
    if not Path(TEST_PDB).exists():
        print(f"ERROR: Test file '{TEST_PDB}' not found. Please download it first.")
    else:
        # 1. Define hardware configuration
        hw_config = HardwareConfig(use_gpu=False, cpu_threads=2) # Use CPU-only for this test
        
        # 2. Instantiate and run the main agent
        gromacs_agent = GromacsAgent(
            workdir="./test_workdir_m4",
            simulation_id="test_1aki_run",
            hardware_config=hw_config
        )
        
        # For this test, we will only run the first few steps manually to verify wiring
        print("\n--- Testing run_pdb2gmx via agent ---")
        gromacs_agent.agent.run(f"Execute the first step for the PDB file '{TEST_PDB}'.")
        print(f"State after step 1: {gromacs_agent.state.get_slm_context()}")
        
        print("\n--- Testing run_editconf via agent ---")
        gromacs_agent.agent.run("Execute the next step in the workflow.")
        print(f"State after step 2: {gromacs_agent.state.get_slm_context()}")

        print("\nMilestone 4 wiring is conceptually complete and ready for a full pipeline test.")