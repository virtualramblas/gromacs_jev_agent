"""Verification test suite for Milestone 1."""

import os
import json
from src.simple_jev import HardwareConfig, SimpleJevEngine, EnvironmentChecker


def run_tests():
    print("=== Step 1: Testing HardwareConfig Thread Limits ===")
    total_cores = os.cpu_count() or 4
    hw = HardwareConfig(use_gpu=False, cpu_threads=total_cores, reserve_cores_for_slm=2)
    expected_threads = max(1, total_cores - 2)
    assert hw.cpu_threads == expected_threads, f"Expected {expected_threads}, got {hw.cpu_threads}"
    print(f"✓ Correctly capped threads to {hw.cpu_threads} (Host Total: {total_cores})")
    print(f"✓ Generated mdrun flags: {' '.join(hw.to_mdrun_flags())}\n")

    print("=== Step 2: Testing SimpleJevEngine Execution & Artifacts ===")
    engine = SimpleJevEngine(base_workdir="./test_workdir", tail_lines=10)
    
    # Run a test job creating a synthetic artifact
    result = engine.run_job(
        job_id="test_init",
        command=["python3", "-c", "with open('sample.txt', 'w') as f: f.write('hello gromacs')"],
        expected_outputs={"sample": "sample.txt"}
    )
    assert result.success, f"Job failed: {result.stderr_tail}"
    assert "sample" in result.created_outputs
    print(f"✓ Job completed successfully in {result.duration_seconds:.4f}s")
    print(f"✓ SLM Payload summary:\n{json.dumps(result.to_slm_payload(), indent=2)}\n")

    print("=== Step 3: Testing Host Environment Dependencies ===")
    gmx_info = EnvironmentChecker.check_gromacs_installed()
    print(f"GROMACS Status: {gmx_info}")
    ollama_ok = EnvironmentChecker.check_ollama_endpoint()
    print(f"Ollama Endpoint Online: {ollama_ok}\n")

    print("Milestone 1 implementation verified!")


if __name__ == "__main__":
    run_tests()