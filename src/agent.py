"""
Milestone 4 & 5: SmolAgent Integration and End-to-End Self-Healing.
Wires up the full tool library to a CodeAgent instance driven by a local
SLM for high-level pipeline orchestration and error recovery.
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
# 1. Complete SmolAgent Tool Definitions
# =====================================================================

class GromacsPipelineTool(Tool):
    """Base tool to provide shared access to the GROMACS tool library and state."""
    def __init__(
        self,
        tool_library: GromacsToolLibrary,
        state_manager: StateManager,
        mdp_generator: Optional[MDPGenerator] = None
    ):
        super().__init__()
        self.tools = tool_library
        self.state = state_manager
        self.mdp_gen = mdp_generator or MDPGenerator()

class RunPdb2gmxTool(GromacsPipelineTool):
    name: str = "run_pdb2gmx"
    description: str = "Generates topology (.top) and coordinate (.gro) files from a PDB. This is the first step."
    inputs: Dict[str, Any] = {"pdb_file": {"type": "string", "description": "Path to the input .pdb file."}}
    output_type: str = "string"

    def forward(self, pdb_file: str) -> str:
        result = self.tools.run_pdb2gmx(pdb_file=pdb_file)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

class RunEditconfTool(GromacsPipelineTool):
    name: str = "run_editconf"
    description: str = "Defines the simulation box and centers the molecule."
    inputs: Dict[str, Any] = {}
    output_type: str = "string"

    def forward(self) -> str:
        gro_file = self.state.get_latest_artifact("pdb2gmx", "gro")
        if not gro_file: return json.dumps({"status": "failed", "error": "Prerequisite 'processed.gro' not found. Run 'run_pdb2gmx' first."})
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
        if not gro_file or not top_file: return json.dumps({"status": "failed", "error": "Prerequisites not met. Complete previous steps."})
        result = self.tools.run_solvate(boxed_gro=gro_file, top_file=top_file)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

class RunGenionTool(GromacsPipelineTool):
    name: str = "run_genion"
    description: str = "Adds neutralizing ions to the solvated system."
    inputs: Dict[str, Any] = {}
    output_type: str = "string"

    def forward(self) -> str:
        gro_file = self.state.get_latest_artifact("solvate", "gro")
        top_file = self.state.get_latest_artifact("pdb2gmx", "top")
        if not gro_file or not top_file: return json.dumps({"status": "failed", "error": "Prerequisites not met."})
        
        ions_mdp = self.mdp_gen.write_mdp("ions", self.state.workdir / "ions.mdp")
        result = self.tools.run_genion(solvated_gro=gro_file, top_file=top_file, ions_mdp=ions_mdp)
        self.state.record_step_result(result.job_id, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

class RunGromppTool(GromacsPipelineTool):
    name: str = "run_grompp"
    description: str = "Assembles the binary run input file (.tpr) for a major simulation phase (em, nvt, npt, md)."
    inputs: Dict[str, Any] = {"step_name": {"type": "string", "description": "The name of the phase: 'em', 'nvt', 'npt', or 'md'."}}
    output_type: str = "string"

    def forward(self, step_name: str) -> str:
        # Determine input files based on the step
        if step_name == 'em':
            gro_file = self.state.get_latest_artifact("genion", "gro")
        else: # nvt, npt, md steps use the output of the previous phase
            prev_step = self.state.STAGES_ORDER[self.state.STAGES_ORDER.index(step_name) - 1]
            gro_file = self.state.get_latest_artifact(prev_step, "gro")
        
        top_file = self.state.get_latest_artifact("pdb2gmx", "top")
        if not gro_file or not top_file: return json.dumps({"status": "failed", "error": f"Prerequisite .gro or .top file for '{step_name}' not found."})

        mdp_file = self.mdp_gen.write_mdp(step_name, self.state.workdir / f"{step_name}.mdp")
        
        # Check for checkpoint file for continuation
        cpt_file = None
        if step_name in ['npt', 'md']:
             cpt_file = self.state.get_latest_artifact(self.state.STAGES_ORDER[self.state.STAGES_ORDER.index(step_name) - 1], "cpt")

        result = self.tools.run_grompp(step_name=step_name, mdp_file=mdp_file, gro_file=gro_file, top_file=top_file, cpt_file=cpt_file)
        self.state.record_step_result(f"grompp_{step_name}", result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

class RunMdrunTool(GromacsPipelineTool):
    name: str = "run_mdrun"
    description: str = "Executes an MD simulation or minimization step (em, nvt, npt, md)."
    inputs: Dict[str, Any] = {"step_name": {"type": "string", "description": "The name of the simulation step to run: 'em', 'nvt', 'npt', or 'md'."}}
    output_type: str = "string"

    def forward(self, step_name: str) -> str:
        tpr_file = self.state.get_latest_artifact(f"grompp_{step_name}", "tpr")
        if not tpr_file: return json.dumps({"status": "failed", "error": f"Prerequisite '{step_name}.tpr' not found. Run 'run_grompp' first."})
        
        result = self.tools.run_mdrun(step_name=step_name, tpr_file=tpr_file)
        self.state.record_step_result(step_name, result.success, result.created_outputs, None if result.success else result.stderr_tail)
        return json.dumps(result.to_slm_payload())

# =====================================================================
# 2. Main Agent Orchestrator with Self-Healing
# =====================================================================

class GromacsAgent:
    """Orchestrates the GROMACS pipeline with SLM-driven execution and error diagnosis."""
    def __init__(self, workdir: str, simulation_id: str, hardware_config: HardwareConfig, model_id: str = "ollama/llama3:8b"):
        self.jev_engine = SimpleJevEngine(base_workdir=workdir)
        self.tool_library = GromacsToolLibrary(jev_engine=self.jev_engine, hardware_config=hardware_config)
        self.state = StateManager(workdir=workdir, simulation_id=simulation_id)
        self.mdp_gen = MDPGenerator()

        tools = [
            RunPdb2gmxTool(self.tool_library, self.state, self.mdp_gen),
            RunEditconfTool(self.tool_library, self.state, self.mdp_gen),
            RunSolvateTool(self.tool_library, self.state, self.mdp_gen),
            RunGenionTool(self.tool_library, self.state, self.mdp_gen),
            RunGromppTool(self.tool_library, self.state, self.mdp_gen),
            RunMdrunTool(self.tool_library, self.state, self.mdp_gen),
        ]
        
        self.system_prompt = f"""You are GROMACS-GPT, an expert AI that automates MD simulations. Your goal is to run the GROMACS workflow step-by-step. The required sequence is: {', '.join(self.state.STAGES_ORDER)}.
        INSTRUCTIONS:
        1. Read the 'CURRENT STATE' to know the current step.
        2. Execute tools strictly in order. NEVER skip a step.
        3. For 'em', 'nvt', 'npt', and 'md', you must call 'run_grompp' BEFORE 'run_mdrun'.
        4. If a step succeeds, call the next tool. If it fails, STOP and report the error."""

        model = LiteLLMModel(model_id=model_id, system_prompt=self.system_prompt)
        self.agent = CodeAgent(tools=tools, model=model, max_steps=20) # Increased max_steps for full pipeline

def run_pipeline(self, pdb_file: str):
        """Starts and runs the full agentic pipeline."""
        print(f"🚀 Starting GROMACS pipeline for {pdb_file}...")
        self.state.data["input_pdb"] = pdb_file
        self.state.save()

        while self.state.data["status"] not in ["completed", "halted"]:
            context = self.state.get_slm_context()
            user_prompt = (
                f"CURRENT STATE: {json.dumps(context, indent=2)}\n\n"
                f"Based on the state, call the single next tool required to advance the pipeline."
            )
            
            print(f"\n--- Agent Turn (Current Step: {context['current_step']}) ---")
            try:
                response = self.agent.run(user_prompt)
                
                # Check if response is a JSON string from a tool
                if isinstance(response, str):
                    try:
                        response_data = json.loads(response)
                    except Exception:
                        response_data = {"raw_output": response}
                elif isinstance(response, dict):
                    response_data = response
                else:
                    response_data = {"raw_output": str(response)}

                print(f"Agent observation: {response_data}")

                if response_data.get("status") == "failed" or response_data.get("success") is False:
                    self.diagnose_and_halt(response_data)
                    break

            except Exception as e:
                print(f"🚨 Agent execution loop failed unexpectedly: {e}")
                self.diagnose_and_halt({"error": str(e)})
                break
        
        print(f"\n✅ Pipeline finished with status: {self.state.data['status']}")
        if self.state.data['status'] == 'halted':
            errors = self.state.data.get('errors', [])
            final_err = errors[-1] if errors else "Pipeline halted with unknown error"
            print(f"🚨 Final Error: {final_err}")
            
    def diagnose_and_halt(self, failure_payload: dict):
        """Uses the SLM to get a human-readable diagnosis of a failure."""
        print("\n--- ⚠️ Entering Self-Healing/Diagnosis Mode ---")
        error_log = failure_payload.get("error", str(failure_payload))

        diagnostic_prompt = f"""A GROMACS simulation step failed. Here is the error log:
---
{error_log}
---
Based on this error, what is the most likely root cause? Explain it briefly for a scientist."""

        try:
            # Compatible with all recent versions of smolagents Model
            if hasattr(self.agent.model, "__call__"):
                model_output = self.agent.model([{"role": "user", "content": diagnostic_prompt}])
                diagnosis = getattr(model_output, "content", str(model_output))
            else:
                diagnosis = f"Diagnostic fallback for error: {error_log}"
            
            print(f"SLM Diagnosis: {diagnosis}")
            self.state.record_step_result(self.state.data['current_step'], False, error_msg=diagnosis)
        except Exception as e:
            print(f"🚨 Diagnosis failed: {e}")
            self.state.record_step_result(self.state.data['current_step'], False, error_msg=error_log)
