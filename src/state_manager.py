"""
Milestone 3: State Manager and MDP Parameter Generation.
Manages pipeline artifacts, checkpoints, and dynamic .mdp generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


# =====================================================================
# 1. Standard MDP Parameter Templates (JSON Definition)
# =====================================================================

DEFAULT_MDP_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "ions": {
        "integrator": "steep",
        "emtol": "1000.0",
        "emstep": "0.01",
        "nsteps": "50000",
        "nstlist": "1",
        "cutoff-scheme": "Verlet",
        "ns_type": "grid",
        "coulombtype": "cutoff",
        "rcoulomb": "1.0",
        "rvdw": "1.0",
        "pbc": "xyz"
    },
    "em": {
        "integrator": "steep",
        "emtol": "1000.0",
        "emstep": "0.01",
        "nsteps": "50000",
        "nstlist": "1",
        "cutoff-scheme": "Verlet",
        "ns_type": "grid",
        "coulombtype": "PME",
        "rcoulomb": "1.0",
        "rvdw": "1.0",
        "pbc": "xyz"
    },
    "nvt": {
        "title": "NVT Equilibration",
        "define": "-DPOSRES",
        "integrator": "md",
        "dt": "0.002",
        "nsteps": "50000",           # 100 ps
        "nstxout": "500",
        "nstvout": "500",
        "nstenergy": "500",
        "nstlog": "500",
        "continuation": "no",
        "constraint_algorithm": "lincs",
        "constraints": "h-bonds",
        "cutoff-scheme": "Verlet",
        "coulombtype": "PME",
        "rcoulomb": "1.0",
        "rvdw": "1.0",
        "tcoupl": "V-rescale",
        "tc-grps": "Protein Non-Protein",
        "tau_t": "0.1 0.1",
        "ref_t": "300 300",
        "pcoupl": "no",
        "pbc": "xyz",
        "gen_vel": "yes",
        "gen_temp": "300",
        "gen_seed": "-1"
    },
    "npt": {
        "title": "NPT Equilibration",
        "define": "-DPOSRES",
        "integrator": "md",
        "dt": "0.002",
        "nsteps": "50000",           # 100 ps
        "nstxout": "500",
        "nstvout": "500",
        "nstenergy": "500",
        "nstlog": "500",
        "continuation": "yes",
        "constraint_algorithm": "lincs",
        "constraints": "h-bonds",
        "cutoff-scheme": "Verlet",
        "coulombtype": "PME",
        "rcoulomb": "1.0",
        "rvdw": "1.0",
        "tcoupl": "V-rescale",
        "tc-grps": "Protein Non-Protein",
        "tau_t": "0.1 0.1",
        "ref_t": "300 300",
        "pcoupl": "Parrinello-Rahman",
        "pcoupltype": "isotropic",
        "tau_p": "2.0",
        "ref_p": "1.0",
        "compressibility": "4.5e-5",
        "refcoord_scaling": "com",
        "pbc": "xyz",
        "gen_vel": "no"
    },
    "md": {
        "title": "Production MD Simulation",
        "integrator": "md",
        "dt": "0.002",
        "nsteps": "500000",          # 1 ns (extendable)
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstenergy": "5000",
        "nstlog": "5000",
        "nstxout-compressed": "5000",
        "compressed-x-grps": "System",
        "continuation": "yes",
        "constraint_algorithm": "lincs",
        "constraints": "h-bonds",
        "cutoff-scheme": "Verlet",
        "coulombtype": "PME",
        "rcoulomb": "1.0",
        "rvdw": "1.0",
        "tcoupl": "V-rescale",
        "tc-grps": "Protein Non-Protein",
        "tau_t": "0.1 0.1",
        "ref_t": "300 300",
        "pcoupl": "Parrinello-Rahman",
        "pcoupltype": "isotropic",
        "tau_p": "2.0",
        "ref_p": "1.0",
        "compressibility": "4.5e-5",
        "pbc": "xyz",
        "gen_vel": "no"
    }
}


# =====================================================================
# 2. Dynamic MDP File Generator
# =====================================================================

class MDPGenerator:
    """Generates standard GROMACS .mdp flat files from JSON configurations."""

    def __init__(self, templates: Optional[Dict[str, Dict[str, Any]]] = None):
        self.templates = templates or DEFAULT_MDP_TEMPLATES

    def write_mdp(
        self,
        step_name: str,
        output_filepath: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesizes a valid .mdp file by combining default templates with runtime overrides.
        
        Args:
            step_name: Stage key ('ions', 'em', 'nvt', 'npt', 'md').
            output_filepath: Target destination for the generated .mdp file.
            overrides: Optional key-value pairs to modify (e.g. {"ref_t": "310 310"}).
        """
        if step_name not in self.templates:
            raise KeyError(f"Unknown MDP stage '{step_name}'. Available: {list(self.templates.keys())}")

        params = dict(self.templates[step_name])
        if overrides:
            params.update(overrides)

        out_path = Path(output_filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"; --------------------------------------------------",
            f"; Auto-generated GROMACS MDP: {step_name.upper()}",
            f"; Generated on: {datetime.utcnow().isoformat()}",
            f"; --------------------------------------------------"
        ]
        for key, value in params.items():
            lines.append(f"{key:<24} = {value}")

        out_path.write_text("\n".join(lines) + "\n")
        return str(out_path)


# =====================================================================
# 3. Simulation State Manager
# =====================================================================

class StateManager:
    """Maintains state_registry.json tracking artifact progression and run status."""

    STAGES_ORDER = [
        "pdb2gmx",
        "editconf",
        "solvate",
        "genion",
        "em",
        "nvt",
        "npt",
        "md"
    ]

    def __init__(self, workdir: str = "./workdir", simulation_id: str = "gmx_sim_001"):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.workdir / "state_registry.json"
        self.simulation_id = simulation_id
        self._init_or_load()

    def _init_or_load(self):
        """Initializes a new registry file if not present; otherwise loads existing."""
        if not self.registry_file.exists():
            self.data = {
                "simulation_id": self.simulation_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "current_step": self.STAGES_ORDER[0],
                "status": "initialized",
                "artifacts": {},
                "errors": []
            }
            self.save()
        else:
            with open(self.registry_file, "r") as f:
                self.data = json.load(f)

    def save(self):
        """Persists current state to JSON atomically."""
        self.data["updated_at"] = datetime.utcnow().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_step_result(
        self,
        step_name: str,
        success: bool,
        output_files: Optional[Dict[str, str]] = None,
        error_msg: Optional[str] = None
    ):
        """Records the outcome of a JevJob execution and updates current pipeline pointer."""
        status_str = "success" if success else "failed"
        self.data["artifacts"][step_name] = {
            "status": status_str,
            "timestamp": datetime.utcnow().isoformat(),
            "files": output_files or {}
        }

        if success:
            self.data["status"] = "in_progress"
            # Advance to next step if available
            try:
                current_idx = self.STAGES_ORDER.index(step_name)
                if current_idx + 1 < len(self.STAGES_ORDER):
                    self.data["current_step"] = self.STAGES_ORDER[current_idx + 1]
                else:
                    self.data["current_step"] = "completed"
                    self.data["status"] = "completed"
            except ValueError:
                pass
        else:
            self.data["status"] = "halted"
            if error_msg:
                self.data["errors"].append({
                    "step": step_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": error_msg
                })

        self.save()

    def get_latest_artifact(self, step_name: str, file_key: str) -> Optional[str]:
        """Retrieves path of an artifact produced in a previous step."""
        step_data = self.data["artifacts"].get(step_name)
        if step_data and step_data.get("status") == "success":
            return step_data["files"].get(file_key)
        return None

    def get_slm_context(self) -> Dict[str, Any]:
        """Returns concise state summary payload for local SLM reasoning."""
        return {
            "simulation_id": self.data["simulation_id"],
            "current_step": self.data["current_step"],
            "status": self.data["status"],
            "completed_steps": [
                s for s, v in self.data["artifacts"].items() if v.get("status") == "success"
            ],
            "last_error": self.data["errors"][-1] if self.data["errors"] else None
        }

    def reset(self):
        """Wipes current registry data and resets to initial stage."""
        self.data = {
            "simulation_id": self.simulation_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "current_step": self.STAGES_ORDER[0],
            "status": "initialized",
            "artifacts": {},
            "errors": []
        }
        self.save()