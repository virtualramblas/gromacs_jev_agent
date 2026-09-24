"""Verification test suite for Milestone 3 State Manager and MDP Generator."""

import json
from pathlib import Path
from src.state_manager import MDPGenerator, StateManager


def run_tests():
    test_dir = Path("./test_workdir_m3")
    test_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Testing MDPGenerator Synthesis & Overrides ===")
    gen = MDPGenerator()
    em_mdp = test_dir / "em.mdp"
    # Write EM with modified step count override
    gen.write_mdp("em", str(em_mdp), overrides={"nsteps": "20000"})
    
    content = em_mdp.read_text()
    assert "integrator               = steep" in content
    assert "nsteps                   = 20000" in content
    print("✓ Successfully generated em.mdp with runtime overrides.")

    print("\n=== Step 2: Testing StateManager Lifecycle & Tracking ===")
    sm = StateManager(workdir=str(test_dir), simulation_id="test_run_101")
    
    # Assert initial state
    assert sm.data["current_step"] == "pdb2gmx"
    assert sm.data["status"] == "initialized"
    print("✓ State initialized at step 'pdb2gmx'.")

    # Simulate Step 1 completion
    sm.record_step_result(
        step_name="pdb2gmx",
        success=True,
        output_files={
            "gro": str(test_dir / "processed.gro"),
            "top": str(test_dir / "topol.top")
        }
    )
    assert sm.data["current_step"] == "editconf"
    assert sm.get_latest_artifact("pdb2gmx", "gro") == str(test_dir / "processed.gro")
    print("✓ Successfully recorded pdb2gmx step and transitioned to 'editconf'.")

    # Verify SLM Context Summary
    slm_context = sm.get_slm_context()
    print(f"✓ SLM Context Payload:\n{json.dumps(slm_context, indent=2)}")

    print("\nMilestone 3 implementation verified!")


if __name__ == "__main__":
    run_tests()