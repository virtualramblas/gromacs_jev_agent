"""
Milestone 1: Core Simple-Jev Engine and Hardware Configuration.
Provides deterministic subprocess sandboxing and resource management for GROMACS workflows.
"""

import os
import shutil
import subprocess
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


# =====================================================================
# 1. Hardware Configuration & Resource Safeguards
# =====================================================================

@dataclass
class HardwareConfig:
    """Manages CPU/GPU resource allocation, safeguarding local SLM headroom."""
    use_gpu: bool = False
    cpu_threads: int = 4
    gpu_id: str = "0"
    reserve_cores_for_slm: int = 2

    def __post_init__(self):
        total_cores = os.cpu_count() or 4
        # Guard: Ensure local SLM processes have unblocked CPU cores
        max_allowed_threads = max(1, total_cores - self.reserve_cores_for_slm)
        if self.cpu_threads > max_allowed_threads:
            self.cpu_threads = max_allowed_threads

    def to_mdrun_flags(self) -> List[str]:
        """Translates hardware configuration into GROMACS mdrun command flags."""
        flags = [
            "-ntomp", str(self.cpu_threads),
            "-pin", "on"
        ]
        if self.use_gpu:
            flags.extend([
                "-nb", "gpu",
                "-pme", "gpu",
                "-bonded", "gpu",
                "-gpu_id", str(self.gpu_id)
            ])
        else:
            flags.extend([
                "-nb", "cpu",
                "-pme", "cpu",
                "-bonded", "cpu"
            ])
        return flags

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. Simple-Jev Data Models & Execution Engine
# =====================================================================

@dataclass
class JevJobResult:
    """Standardized result envelope returned by Simple-Jev executions."""
    job_id: str
    command: List[str]
    success: bool
    exit_code: int
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    expected_outputs: Dict[str, str]
    created_outputs: Dict[str, str] = field(default_factory=dict)
    missing_outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_slm_payload(self) -> Dict[str, Any]:
        """Compact summary payload optimized for consumption by a local SLM."""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_seconds, 2),
            "created_files": list(self.created_outputs.keys()),
            "missing_files": self.missing_outputs,
            "error_summary": self.stderr_tail if not self.success else ""
        }


class SimpleJevEngine:
    """
    Simple Job Execution Vehicle (Simple-Jev).
    Executes tasks in sandboxed working directories with output capture and validation.
    """

    def __init__(self, base_workdir: str = "./workdir", tail_lines: int = 25):
        self.base_workdir = Path(base_workdir).resolve()
        self.tail_lines = tail_lines
        self.base_workdir.mkdir(parents=True, exist_ok=True)

    def _truncate_log(self, text: str) -> str:
        """Extracts the last N lines of a log to preserve SLM context windows."""
        if not text:
            return ""
        lines = text.strip().splitlines()
        truncated = lines[-self.tail_lines:]
        return "\n".join(truncated)

    def run_job(
        self,
        job_id: str,
        command: List[str],
        expected_outputs: Optional[Dict[str, str]] = None,
        stdin_input: Optional[str] = None,
        sub_dir: Optional[str] = None
    ) -> JevJobResult:
        """
        Executes a command inside the designated working directory.
        
        Args:
            job_id: Unique label for the job stage.
            command: Command argument list.
            expected_outputs: Dict mapping aliases to relative file names (e.g. {"gro": "out.gro"}).
            stdin_input: Text to pass to process stdin (for non-interactive selection).
            sub_dir: Optional sub-folder inside base_workdir.
        """
        target_dir = self.base_workdir / sub_dir if sub_dir else self.base_workdir
        target_dir.mkdir(parents=True, exist_ok=True)

        expected = expected_outputs or {}
        start_time = time.time()

        try:
            process = subprocess.run(
                command,
                cwd=str(target_dir),
                input=stdin_input,
                capture_output=True,
                text=True,
                check=False
            )
            duration = time.time() - start_time
            exit_code = process.returncode
            stdout_tail = self._truncate_log(process.stdout)
            stderr_tail = self._truncate_log(process.stderr)
        except Exception as exc:
            duration = time.time() - start_time
            return JevJobResult(
                job_id=job_id,
                command=command,
                success=False,
                exit_code=-1,
                duration_seconds=duration,
                stdout_tail="",
                stderr_tail=str(exc),
                expected_outputs=expected,
                missing_outputs=list(expected.values())
            )

        # Artifact validation: check that required files actually exist
        created = {}
        missing = []
        for label, filename in expected.items():
            file_path = target_dir / filename
            if file_path.exists() and file_path.stat().st_size > 0:
                created[label] = str(file_path)
            else:
                missing.append(filename)

        # A job is successful ONLY if exit code is 0 and all expected artifacts are present
        success = (exit_code == 0) and (len(missing) == 0)

        return JevJobResult(
            job_id=job_id,
            command=command,
            success=success,
            exit_code=exit_code,
            duration_seconds=duration,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            expected_outputs=expected,
            created_outputs=created,
            missing_outputs=missing
        )


# =====================================================================
# 3. Environment & Prerequisite Verification
# =====================================================================

class EnvironmentChecker:
    """Verifies host dependencies before launching agentic workflows."""

    @staticmethod
    def check_gromacs_installed() -> Dict[str, Any]:
        """Checks whether 'gmx' is in PATH and captures version string."""
        gmx_path = shutil.which("gmx")
        if not gmx_path:
            return {"installed": False, "version": None, "path": None}

        try:
            proc = subprocess.run(["gmx", "--version"], capture_output=True, text=True, check=True)
            for line in proc.stdout.splitlines():
                if "GROMACS version" in line:
                    return {"installed": True, "version": line.strip(), "path": gmx_path}
            return {"installed": True, "version": "Unknown Version", "path": gmx_path}
        except Exception as exc:
            return {"installed": False, "error": str(exc), "path": gmx_path}

    @staticmethod
    def check_ollama_endpoint(host: str = "http://localhost:11434") -> bool:
        """Pings local Ollama server to ensure SLM connectivity."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False