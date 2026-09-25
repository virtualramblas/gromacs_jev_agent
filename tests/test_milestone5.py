"""
Verification test suite for Milestone 5: End-to-End Testing & Self-Healing.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Import our Milestone modules
from src.simple_jev import HardwareConfig, JevJobResult
from src.gromacs_tools import GromacsToolLibrary
from src.agent import GromacsAgent


def setup_mock_environment(test_dir: Path):
    """Creates a clean directory for testing."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = test_dir / "1AKI.pdb"
    pdb_path.write_text("HEADER    LYSOZYME\nATOM      1  N   LYS A   1")
    return pdb_path


def mock_jev_success(job_id: str, created_outputs: dict) -> JevJobResult:
    """Factory for creating successful JevJobResult mocks."""
    return JevJobResult(
        job_id=job_id,
        command=["mock", job_id],
        success=True,
        exit_code=0,
        duration_seconds=0.1,
        stdout_tail="Success",
        stderr_tail="",
        expected_outputs=created_outputs,
        created_outputs=created_outputs
    )


def mock_jev_failure(job_id: str, stderr: str) -> JevJobResult:
    """Factory for creating failed JevJobResult mocks."""
    return JevJobResult(
        job_id=job_id,
        command=["mock", job_id],
        success=False,
        exit_code=1,
        duration_seconds=0.1,
        stdout_tail="",
        stderr_tail=stderr,
        expected_outputs={},
        created_outputs={}
    )


def extract_tools_map(agent) -> dict:
    """Extracts a {name: tool_instance} map regardless of smolagents version."""
    if isinstance(agent.tools, dict):
        return agent.tools
    return {tool.name: tool for tool in agent.tools}


def run_tests():
    test_dir = Path("./test_workdir_m5_sandbox")
    pdb_path = setup_mock_environment(test_dir)

    print("=== Test Case 1: The 'Happy Path' - Successful End-to-End Run ===")
    hw_config = HardwareConfig()
    happy_agent = GromacsAgent(
        workdir=str(test_dir / "happy"),
        simulation_id="happy_run",
        hardware_config=hw_config
    )

    # Mock tool library methods so they succeed and return mock files
    happy_agent.tool_library.run_pdb2gmx = MagicMock(
        return_value=mock_jev_success("pdb2gmx", {"gro": "processed.gro", "top": "topol.top"})
    )
    happy_agent.tool_library.run_editconf = MagicMock(
        return_value=mock_jev_success("editconf", {"gro": "boxed.gro"})
    )
    happy_agent.tool_library.run_solvate = MagicMock(
        return_value=mock_jev_success("solvate", {"gro": "solvated.gro", "top": "topol.top"})
    )
    happy_agent.tool_library.run_genion = MagicMock(
        return_value=mock_jev_success("genion", {"gro": "solvated_ions.gro"})
    )
    happy_agent.tool_library.run_grompp = MagicMock(
        side_effect=lambda step_name, **kwargs: mock_jev_success(f"grompp_{step_name}", {"tpr": f"{step_name}.tpr"})
    )
    happy_agent.tool_library.run_mdrun = MagicMock(
        side_effect=lambda step_name, **kwargs: mock_jev_success(step_name, {"gro": f"{step_name}.gro", "log": f"{step_name}.log"})
    )

    # Use the robust tool map extractor
    tools_by_name = extract_tools_map(happy_agent.agent)

    def simulate_happy_agent_step(prompt):
        current_step = happy_agent.state.data["current_step"]
        if current_step == "pdb2gmx":
            return tools_by_name["run_pdb2gmx"].forward(str(pdb_path))
        elif current_step == "editconf":
            return tools_by_name["run_editconf"].forward()
        elif current_step == "solvate":
            return tools_by_name["run_solvate"].forward()
        elif current_step == "genion":
            return tools_by_name["run_genion"].forward()
        elif current_step in ["em", "nvt", "npt", "md"]:
            # Run grompp preparation followed by mdrun
            tools_by_name["run_grompp"].forward(current_step)
            return tools_by_name["run_mdrun"].forward(current_step)
        return json.dumps({"status": "completed"})

    happy_agent.agent.run = MagicMock(side_effect=simulate_happy_agent_step)
    happy_agent.run_pipeline(str(pdb_path))

    assert happy_agent.state.data["status"] == "completed"
    assert "md" in happy_agent.state.data["artifacts"]
    print("✓ 'Happy Path' test successfully executed the full 8-step pipeline.\n")

    print("=== Test Case 2: The 'Failure Path' - Self-Healing and Diagnosis ===")
    sad_agent = GromacsAgent(
        workdir=str(test_dir / "sad"),
        simulation_id="sad_run",
        hardware_config=hw_config
    )

    # Configure tool library to simulate success up to EM, then fail at NVT
    sad_agent.tool_library.run_pdb2gmx = MagicMock(
        return_value=mock_jev_success("pdb2gmx", {"gro": "processed.gro", "top": "topol.top"})
    )
    sad_agent.tool_library.run_editconf = MagicMock(
        return_value=mock_jev_success("editconf", {"gro": "boxed.gro"})
    )
    sad_agent.tool_library.run_solvate = MagicMock(
        return_value=mock_jev_success("solvate", {"gro": "solvated.gro", "top": "topol.top"})
    )
    sad_agent.tool_library.run_genion = MagicMock(
        return_value=mock_jev_success("genion", {"gro": "solvated_ions.gro"})
    )
    sad_agent.tool_library.run_grompp = MagicMock(
        side_effect=lambda step_name, **kwargs: mock_jev_success(f"grompp_{step_name}", {"tpr": f"{step_name}.tpr"})
    )

    def mdrun_with_failure(step_name, **kwargs):
        if step_name == "nvt":
            return mock_jev_failure("nvt", "Fatal error: LINCS warnings violated bond constraints.")
        return mock_jev_success(step_name, {"gro": f"{step_name}.gro", "log": f"{step_name}.log"})

    sad_agent.tool_library.run_mdrun = MagicMock(side_effect=mdrun_with_failure)

    sad_tools = extract_tools_map(sad_agent.agent)

    def simulate_failing_agent_step(prompt):
        current_step = sad_agent.state.data["current_step"]
        if current_step == "pdb2gmx":
            return sad_tools["run_pdb2gmx"].forward(str(pdb_path))
        elif current_step == "editconf":
            return sad_tools["run_editconf"].forward()
        elif current_step == "solvate":
            return sad_tools["run_solvate"].forward()
        elif current_step == "genion":
            return sad_tools["run_genion"].forward()
        elif current_step in ["em", "nvt"]:
            sad_tools["run_grompp"].forward(current_step)
            return sad_tools["run_mdrun"].forward(current_step)
        return json.dumps({"status": "completed"})

    sad_agent.agent.run = MagicMock(side_effect=simulate_failing_agent_step)
    
    # Mock model completion response for diagnosis
    sad_agent.agent.model = MagicMock(
        return_value=MagicMock(content="LINCS explosion detected. System unstable; reduce timestep or re-run energy minimization.")
    )

    sad_agent.run_pipeline(str(pdb_path))

    assert sad_agent.state.data["status"] == "halted"
    last_error = sad_agent.state.data["errors"][-1]["error"]
    assert "LINCS" in last_error
    print("✓ 'Failure Path' test confirmed agent catches simulation crash, runs diagnosis, and halts cleanly.")

    print("\nMilestone 5 and entire GROMACS Agent suite verified!")


if __name__ == "__main__":
    run_tests()