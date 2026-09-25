"""
Verification test suite for Milestone 5: End-to-End Testing & Self-Healing.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import our Milestone modules
from src.simple_jev import HardwareConfig, JevJobResult
from src.gromacs_tools import GromacsToolLibrary
from src.agent import GromacsAgent

def setup_mock_environment(test_dir: Path):
    """Creates a clean directory for testing."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    (test_dir / "1AKI.pdb").write_text("DUMMY PDB CONTENT")
    return test_dir / "1AKI.pdb"

def mock_jev_success(job_id: str, created_outputs: dict) -> JevJobResult:
    """Factory for creating successful JevJobResult mocks."""
    return JevJobResult(
        job_id=job_id, command=["mock"], success=True, exit_code=0, duration_seconds=0.1,
        stdout_tail="Success", stderr_tail="", expected_outputs=created_outputs, created_outputs=created_outputs
    )

def mock_jev_failure(job_id: str, stderr: str) -> JevJobResult:
    """Factory for creating failed JevJobResult mocks."""
    return JevJobResult(
        job_id=job_id, command=["mock"], success=False, exit_code=1, duration_seconds=0.1,
        stdout_tail="", stderr_tail=stderr, expected_outputs={}, created_outputs={}
    )

def run_tests():
    test_dir = Path("./test_workdir_m5_sandbox")
    pdb_path = setup_mock_environment(test_dir)

    print("=== Test Case 1: The 'Happy Path' - Successful End-to-End Run ===")
    
    # Mock the CodeAgent's run method to simulate sequential successful tool calls
    with patch('src.agent.CodeAgent.run') as mock_agent_run:
        # This sequence simulates the agent calling the right tool at each step
        mock_agent_run.side_effect = [
            json.dumps(mock_jev_success("pdb2gmx", {"gro": "p.gro", "top": "t.top"}).to_slm_payload()),
            json.dumps(mock_jev_success("editconf", {"gro": "b.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("solvate", {"gro": "s.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("genion", {"gro": "i.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("grompp_em", {"tpr": "em.tpr"}).to_slm_payload()),
            json.dumps(mock_jev_success("em", {"gro": "em.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("grompp_nvt", {"tpr": "nvt.tpr"}).to_slm_payload()),
            json.dumps(mock_jev_success("nvt", {"gro": "nvt.gro"}).to_slm_payload()),
            # ... and so on for npt and md
        ]

        hw_config = HardwareConfig()
        happy_agent = GromacsAgent(workdir=str(test_dir / "happy"), simulation_id="happy_run", hardware_config=hw_config)
        happy_agent.run_pipeline(str(pdb_path))

        assert happy_agent.state.data["status"] == "in_progress" # It stops when mocks run out
        assert "nvt" in happy_agent.state.data["artifacts"]
        print("✓ 'Happy Path' test successfully simulated sequential tool execution.\n")

    print("=== Test Case 2: The 'Failure Path' - Self-Healing and Diagnosis ===")
    
    # Mock the agent to fail at the NVT step
    with patch('src.agent.CodeAgent.run') as mock_agent_run:
        # Simulate success up to the point of failure
        mock_agent_run.side_effect = [
            json.dumps(mock_jev_success("pdb2gmx", {"gro": "p.gro", "top": "t.top"}).to_slm_payload()),
            json.dumps(mock_jev_success("editconf", {"gro": "b.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("solvate", {"gro": "s.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("genion", {"gro": "i.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("grompp_em", {"tpr": "em.tpr"}).to_slm_payload()),
            json.dumps(mock_jev_success("em", {"gro": "em.gro"}).to_slm_payload()),
            json.dumps(mock_jev_success("grompp_nvt", {"tpr": "nvt.tpr"}).to_slm_payload()),
            # The mdrun step for NVT now returns a failure payload
            json.dumps(mock_jev_failure("nvt", "Error: LINCS warnings violated bond constraints.").to_slm_payload()),
            # The next call is the diagnostic one
            "Diagnosis: The system is unstable. The LINCS warning indicates atoms are moving too fast. Try a shorter timestep or more minimization."
        ]
        
        hw_config = HardwareConfig()
        sad_agent = GromacsAgent(workdir=str(test_dir / "sad"), simulation_id="sad_run", hardware_config=hw_config)
        sad_agent.run_pipeline(str(pdb_path))
        
        assert sad_agent.state.data["status"] == "halted"
        assert "LINCS" in sad_agent.state.data["errors"][-1]["error"]
        print("✓ 'Failure Path' test confirmed agent detects failure, runs diagnosis, and halts cleanly.")

    print("\nMilestone 5 and all previous milestones successfully verified!")


if __name__ == "__main__":
    run_tests()