# gromacs_jev_agent

An agentic workflow for [GROMACS](https://www.gromacs.org/) molecular dynamics simulations, built around **Simple-Jev** (Simple Job Execution Vehicle), a deterministic, sandboxed job runner designed to work with a local Small Language Model (SLM) served by [Ollama](https://ollama.com/). Other providers to serve local models would be supported next.

> **Status:** early development. The final implementation would be merged (or replace) the [gromacs_smolagent_2](https://github.com/virtualramblas/gromacs_smolagent_2) backend.

## Overview

An LLM-driven agent should decide *what* to run in a GROMACS workflow, but it should not be trusted to run commands or judge success on its own. Simple-Jev separates the two concerns:

- **Deterministic execution.** Each job runs as a subprocess in its own working directory, with captured output and an explicit exit code.
- **Artifact validation.** A job counts as successful only if the exit code is `0` **and** every expected output file exists and is non-empty.
- **SLM-friendly results.** Logs are truncated to the last N lines, and each result can be turned into a compact payload that fits a small model's context window.
- **Resource safeguards.** CPU thread counts are capped so the local SLM always keeps free cores.

## Components

### `HardwareConfig`

Manages CPU/GPU allocation and translates it into `gmx mdrun` flags.

| Field | Default | Description |
| --- | --- | --- |
| `use_gpu` | `False` | Offload non-bonded, PME and bonded work to the GPU |
| `cpu_threads` | `4` | OpenMP threads (`-ntomp`) |
| `gpu_id` | `"0"` | GPU device ID (`-gpu_id`) |
| `reserve_cores_for_slm` | `2` | Cores kept free for the local SLM; `cpu_threads` is automatically capped at `total_cores - reserve_cores_for_slm` (minimum 1) |

`to_mdrun_flags()` returns the matching `mdrun` flags (always `-ntomp` and `-pin on`, plus `-nb`, `-pme` and `-bonded` set to `gpu` or `cpu`).

### `SimpleJevEngine`

Runs a command inside a sandboxed working directory (`base_workdir`, optionally with a sub-folder per stage) and returns a `JevJobResult`.

`run_job()` arguments:

| Argument | Description |
| --- | --- |
| `job_id` | Unique label for the job stage |
| `command` | Command as an argument list |
| `expected_outputs` | Mapping of alias to relative file name, e.g. `{"gro": "out.gro"}` |
| `stdin_input` | Text passed to stdin, for non-interactive selections |
| `sub_dir` | Optional sub-folder inside `base_workdir` |

### `JevJobResult`

Standardized result envelope: job ID, command, success flag, exit code, duration, stdout/stderr tails, and which expected outputs were created or missing. Use `to_dict()` for the full record, or `to_slm_payload()` for a compact summary (success, exit code, duration, created and missing files, and the stderr tail on failure).

### `EnvironmentChecker`

Verifies prerequisites before a workflow starts:

- `check_gromacs_installed()` checks that `gmx` is on `PATH` and reports its version.
- `check_ollama_endpoint()` pings the local Ollama server (default `http://localhost:11434`).

## Requirements

- Python 3.12+
- [GROMACS](https://manual.gromacs.org/current/install-guide/index.html) with `gmx` on your `PATH`
- [Ollama](https://ollama.com/) running locally, with an SLM pulled (Qwen 2.5 Coder 1.5B is currently the best compromise between speed and quality)

## Installation

```bash
git clone https://github.com/virtualramblas/gromacs_jev_agent.git
cd gromacs_jev_agent

pip install -r requirements.txt
```

## Usage

To start the workflow:  
```bash
python run_simulation.py <path_to_pdb_file> --model_id ollama/<model_id>
```

<!-- TODO: add an end-to-end example once the agent layer is in place, and verify the snippet above against your inputs. -->

## Repository structure

```
gromacs_jev_agent/
├── src/                # Source code
├── tests/              # Test suite
├── LICENSE             # Apache-2.0
└── README.md
```

## Testing

```bash
pytest tests/
```


## Limitations

- Success checks confirm that output files exist and are non-empty, not that the simulation results are scientifically valid.

## Contributing

Issues and pull requests are welcome. <!-- TODO: add guidelines if you want them -->

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgments

- [GROMACS](https://www.gromacs.org/) and its developers
- [Ollama](https://ollama.com/)