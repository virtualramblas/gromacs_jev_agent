#!/usr/bin/env python3
"""
Parameterized GROMACS Agent CLI Launcher.
Runs the autonomous agentic GROMACS pipeline on a real PDB file with custom hardware tuning.
"""

import sys
import argparse
from pathlib import Path

from src.simple_jev import HardwareConfig, EnvironmentChecker
from src.agent import GromacsAgent


def parse_arguments():
    """Parses command-line arguments, setting default values for optional parameters."""
    parser = argparse.ArgumentParser(
        description="✨ GROMACS-GPT: Autonomous local Agentic MD Simulation Orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1. Mandatory Arguments (Positional)
    parser.add_argument(
        "pdb_target",
        type=str,
        help="Path to the target input macromolecule .pdb file (Mandatory)."
    )

    # 2. Optional Hardware Resource Arguments
    parser.add_argument(
        "--use_gpu",
        action="store_true",  # Defaults to False, activates if flag is provided
        help="Enable CUDA/GPU offloading acceleration inside GROMACS."
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=4,
        help="Number of OpenMP CPU threads to allocate to GROMACS mdrun."
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="Select the specific GPU device ID to bind for CUDA offloading."
    )
    parser.add_argument(
        "--reserve_cores_for_slm",
        type=int,
        default=2,
        help="Number of host CPU cores to protect and reserve exclusively for your local SLM."
    )

    # 3. Optional Workspace and Model Arguments
    parser.add_argument(
        "--workdir",
        type=str,
        default="./real_simulation_workdir",
        help="Target workspace folder for simulation logs, artifacts, and outputs."
    )
    parser.add_argument(
        "--simulation_id",
        type=str,
        default="gmx_lysozyme_run",
        help="Unique identifier label for the simulation tracking state."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="ollama/llama3:8b",
        help="Active local Ollama model tag to drive agentic tool calling and error self-healing."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Verify input PDB target exists immediately before starting environment checks
    pdb_path = Path(args.pdb_target)
    if not pdb_path.exists():
        print(f"🚨 File Error: Target PDB file '{args.pdb_target}' does not exist.")
        sys.exit(1)

    print("=== 🔍 Step 1: Validating Local Host Environment ===")
    gmx_info = EnvironmentChecker.check_gromacs_installed()
    ollama_online = EnvironmentChecker.check_ollama_endpoint()

    print(f"• GROMACS: {'✓ Found' if gmx_info['installed'] else '✗ Missing'}")
    if gmx_info['installed']:
        print(f"  Path: {gmx_info['path']}")
        print(f"  Version: {gmx_info['version']}")
        
    print(f"• Ollama:  {'✓ Online' if ollama_online else '✗ Offline (Ensure Ollama service is running)'}")

    if not gmx_info['installed'] or not ollama_online:
        print("\n🚨 Execution Halted: Missing local environment dependencies.")
        sys.exit(1)

    print("\n=== 🖥️ Step 2: Configuring Parameterized Hardware Resources ===")
    # Instantiate HardwareConfig with CLI overrides
    hw_config = HardwareConfig(
        use_gpu=args.use_gpu,
        cpu_threads=args.cpu_threads,
        gpu_id=args.gpu_id,
        reserve_cores_for_slm=args.reserve_cores_for_slm
    )
    print(f"• Allocated {hw_config.cpu_threads} CPU cores to GROMACS.")
    print(f"• GPU Acceleration: {'Enabled (Device ID: ' + hw_config.gpu_id + ')' if hw_config.use_gpu else 'Disabled'}")

    print("\n=== 🤖 Step 3: Initializing GromacsAgent Orchestrator ===")
    gromacs_agent = GromacsAgent(
        workdir=args.workdir,
        simulation_id=args.simulation_id,
        hardware_config=hw_config,
        model_id=args.model_id
    )
    print(f"• Simulation ID: {args.simulation_id}")
    print(f"• Sandbox Workspace: {Path(args.workdir).resolve()}")
    print(f"• Local SLM: {args.model_id}")

    print("\n=== 🏁 Step 4: Initiating Autonomous Production MD ===")
    try:
        gromacs_agent.run_pipeline(str(pdb_path.resolve()))
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation run interrupted by user. State has been safely recorded in state_registry.json.")


if __name__ == "__main__":
    main()