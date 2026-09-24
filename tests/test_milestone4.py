"""
Verification test suite for Milestone 4.
Tests the SmolAgent tool calling logic, state changes, and SLM instructions.
"""

import os
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Import our Milestone modules
from src.simple_jev import HardwareConfig, JevJobResult
from src.gromacs_tools import GromacsToolLibrary
from src.state_manager import StateManager
from src.agent import (
    GromacsAgent,
    RunPdb2gmxTool,
    RunEditconfTool,
    RunSolvateTool
)


def setup_mock_environment(test_dir: Path):
    """Creates directory structures and mock input files for the test."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy PDB file
    pdb_path = test_dir / "1AKI.pdb"
    pdb_path.write_text("HEADER    LYSOZYME\nATOM      1  N   LYS A   1      11.12   9.32  14.20  1.00 15.00           N")
    return pdb_path


def run_tests():
    test_dir = Path("./test_workdir_m4_sandbox")
    pdb_path = setup_mock_environment(test_dir)

    print("=== Step 1: Initializing Orchestrator & State Manager ===")
    hw_config = HardwareConfig(use_gpu=False, cpu_threads=2)
    
    # Initialize our orchestrator agent wrapper
    gromacs_agent = GromacsAgent(
        workdir=str(test_dir),
        simulation_id="test_m4_mock_run",
        hardware_config=hw_config
    )
    
    # Confirm initial state registry looks correct
    context = gromacs_agent.state.get_slm_context()
    assert context["current_step"] == "pdb2gmx"
    assert context["status"] == "initialized"
    print("✓ State registry successfully initialized at 'pdb2gmx'.\n")


    print("=== Step 2: Testing SmolAgent Tool Interfacing (Mocked Execution) ===")
    
    # Mock the GROMACS Tool Library to return immediate mock JevJobResults
    # This prevents the test from actually trying to invoke the 'gmx' shell command
    mock_library = MagicMock(spec=GromacsToolLibrary)
    
    # Mock pdb2gmx result
    mock_library.run_pdb2gmx.return_value = JevJobResult(
        job_id="pdb2gmx",
        command=["gmx", "pdb2gmx", "dummy_args"],
        success=True,
        exit_code=0,
        duration_seconds=0.5,
        stdout_tail="Processed successfully",
        stderr_tail="",
        expected_outputs={"gro": "processed.gro", "top": "topol.top"},
        created_outputs={"gro": str(test_dir / "processed.gro"), "top": str(test_dir / "topol.top")}
    )

    # Re-wire tools in the agent with our mock library
    pdb2gmx_tool = RunPdb2gmxTool(mock_library, gromacs_agent.state)
    editconf_tool = RunEditconfTool(mock_library, gromacs_agent.state)

    # Simulate the PDB file registration
    gromacs_agent.state.data["input_pdb"] = str(pdb_path)
    gromacs_agent.state.save()

    print("→ Executing RunPdb2gmxTool wrapper...")
    response_json = pdb2gmx_tool.forward(pdb_file=str(pdb_path))
    response = json.loads(response_json)
    
    assert response["success"] is True
    assert gromacs_agent.state.data["current_step"] == "editconf"
    assert gromacs_agent.state.data["status"] == "in_progress"
    print("✓ Tool parsed output, mutated state registry, and progressed step to 'editconf'.")


    print("\n=== Step 3: Verifying State Constraints & Missing Prerequisites ===")
    # Create a fresh registry state where previous steps didn't run, and run editconf
    broken_state = StateManager(workdir=str(test_dir / "broken"), simulation_id="broken_run")
    broken_editconf_tool = RunEditconfTool(mock_library, broken_state)
    
    err_response_json = broken_editconf_tool.forward()
    err_response = json.loads(err_response_json)
    assert err_response["status"] == "failed"
    assert "Prerequisite" in err_response["error"]
    print("✓ State Manager successfully blocked out-of-order execution when files were missing.")


    print("\n=== Step 4: Local SLM System Prompt Inspection ===")
    # Inspect the system prompt inside the agent to verify our GROMACS-GPT system instructions are packed
    # Smolagents stores system prompts inside the agent's default prompt templates or system_prompt
    agent_instructions = gromacs_agent.agent.system_prompt
    assert "GROMACS-GPT" in agent_instructions
    assert "NEVER skip a step" in agent_instructions
    print("✓ Agent cognitive framing is verified and contains standard pipeline constraints.")

    print("\nMilestone 4 test suite execution successful!")


if __name__ == "__main__":
    run_tests()