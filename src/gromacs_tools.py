"""
Milestone 2: Deterministic GROMACS Tool Wrappers.
Wraps all standard GROMACS pipeline commands into non-interactive JevJob tasks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.simple_jev import SimpleJevEngine, HardwareConfig, JevJobResult


class GromacsToolLibrary:
    """Library of deterministic GROMACS wrappers executed via Simple-Jev."""

    def __init__(self, jev_engine: SimpleJevEngine, hardware_config: Optional[HardwareConfig] = None):
        self.engine = jev_engine
        self.hardware = hardware_config or HardwareConfig()

    # -------------------------------------------------------------------------
    # Step 1: PDB to GROMACS Topology & Coordinates
    # -------------------------------------------------------------------------
    def run_pdb2gmx(
        self,
        pdb_file: str,
        forcefield: str = "amber99sb-ildn",
        water: str = "tip3p",
        ignh: bool = True
    ) -> JevJobResult:
        """
        Converts input PDB to .gro coordinates and .top topology.
        Explicitly specifies force field and water model to avoid terminal menus.
        """
        output_gro = "processed.gro"
        output_top = "topol.top"

        cmd = [
            "gmx", "pdb2gmx",
            "-f", str(Path(pdb_file).resolve()),
            "-o", output_gro,
            "-p", output_top,
            "-ff", forcefield,
            "-water", water
        ]
        if ignh:
            cmd.append("-ignh")

        return self.engine.run_job(
            job_id="pdb2gmx",
            command=cmd,
            expected_outputs={
                "gro": output_gro,
                "top": output_top
            }
        )

    # -------------------------------------------------------------------------
    # Step 2: Define Unit Cell / Box
    # -------------------------------------------------------------------------
    def run_editconf(
        self,
        gro_file: str,
        box_type: str = "cubic",
        distance_nm: float = 1.0,
        center: bool = True
    ) -> JevJobResult:
        """Centers solute in box and sets minimum clearance distance to box edges."""
        output_gro = "boxed.gro"

        cmd = [
            "gmx", "editconf",
            "-f", str(Path(gro_file).resolve()),
            "-o", output_gro,
            "-bt", box_type,
            "-d", str(distance_nm)
        ]
        if center:
            cmd.append("-c")

        return self.engine.run_job(
            job_id="editconf",
            command=cmd,
            expected_outputs={"gro": output_gro}
        )

    # -------------------------------------------------------------------------
    # Step 3: Solvate the Box
    # -------------------------------------------------------------------------
    def run_solvate(
        self,
        boxed_gro: str,
        top_file: str,
        solvent_box: str = "spc216.gro"
    ) -> JevJobResult:
        """Fills the simulation box with solvent molecules and updates topol.top."""
        output_gro = "solvated.gro"

        cmd = [
            "gmx", "solvate",
            "-cp", str(Path(boxed_gro).resolve()),
            "-cs", solvent_box,
            "-p", str(Path(top_file).resolve()),
            "-o", output_gro
        ]

        return self.engine.run_job(
            job_id="solvate",
            command=cmd,
            expected_outputs={
                "gro": output_gro,
                "top": str(Path(top_file).name)
            }
        )

    # -------------------------------------------------------------------------
    # Step 4: Add Neutralizing Ions
    # -------------------------------------------------------------------------
    def run_genion(
        self,
        solvated_gro: str,
        top_file: str,
        ions_mdp: str,
        solvent_group_idx: str = "13",
        pname: str = "NA",
        nname: str = "CL",
        neutral: bool = True
    ) -> JevJobResult:
        """
        Assembles ion .tpr with grompp, then neutralizes the net charge
        by piping group selection (e.g. '13' for SOL) into gmx genion.
        """
        temp_tpr = "ions_temp.tpr"
        output_gro = "solvated_ions.gro"

        # 4a. Compile temporary TPR for genion
        grompp_cmd = [
            "gmx", "grompp",
            "-f", str(Path(ions_mdp).resolve()),
            "-c", str(Path(solvated_gro).resolve()),
            "-p", str(Path(top_file).resolve()),
            "-o", temp_tpr,
            "-maxwarn", "2"
        ]
        grompp_res = self.engine.run_job(
            job_id="genion_grompp",
            command=grompp_cmd,
            expected_outputs={"tpr": temp_tpr}
        )
        if not grompp_res.success:
            return grompp_res

        # 4b. Run genion with piped group index
        genion_cmd = [
            "gmx", "genion",
            "-s", temp_tpr,
            "-p", str(Path(top_file).resolve()),
            "-o", output_gro,
            "-pname", pname,
            "-nname", nname
        ]
        if neutral:
            genion_cmd.append("-neutral")

        return self.engine.run_job(
            job_id="genion",
            command=genion_cmd,
            expected_outputs={"gro": output_gro},
            stdin_input=f"{solvent_group_idx}\n"
        )

    # -------------------------------------------------------------------------
    # Step 5: Preprocessing (grompp) for EM / NVT / NPT / Production
    # -------------------------------------------------------------------------
    def run_grompp(
        self,
        step_name: str,
        mdp_file: str,
        gro_file: str,
        top_file: str,
        cpt_file: Optional[str] = None
    ) -> JevJobResult:
        """Assembles final simulation input (.tpr) for a given run phase."""
        output_tpr = f"{step_name}.tpr"

        cmd = [
            "gmx", "grompp",
            "-f", str(Path(mdp_file).resolve()),
            "-c", str(Path(gro_file).resolve()),
            "-p", str(Path(top_file).resolve()),
            "-o", output_tpr,
            "-maxwarn", "2"
        ]
        if cpt_file:
            cmd.extend(["-t", str(Path(cpt_file).resolve())])

        return self.engine.run_job(
            job_id=f"grompp_{step_name}",
            command=cmd,
            expected_outputs={"tpr": output_tpr}
        )

    # -------------------------------------------------------------------------
    # Step 6: Simulation Execution (mdrun) with Hardware Injection
    # -------------------------------------------------------------------------
    def run_mdrun(
        self,
        step_name: str,
        tpr_file: str
    ) -> JevJobResult:
        """
        Executes GROMACS mdrun, dynamically injecting CPU/GPU configuration
        and thread reservation rules established in HardwareConfig.
        """
        cmd = [
            "gmx", "mdrun",
            "-deffnm", step_name,
            "-s", str(Path(tpr_file).resolve())
        ]
        # Attach parameterized hardware flags (-ntomp, -pin, -nb gpu, etc.)
        cmd.extend(self.hardware.to_mdrun_flags())

        expected = {
            "gro": f"{step_name}.gro",
            "log": f"{step_name}.log"
        }
        # Production & Equilibration runs also produce trajectory and checkpoint
        if step_name != "em":
            expected["cpt"] = f"{step_name}.cpt"
            expected["xtc"] = f"{step_name}.xtc"

        return self.engine.run_job(
            job_id=f"mdrun_{step_name}",
            command=cmd,
            expected_outputs=expected
        )