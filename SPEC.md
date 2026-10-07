# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Runs gpu-bench-jax-xla-forwardpass.py (the same script as 226/426): a small JAX/XLA transformer forward pass on synthetic tokens. dtype, sequence_len, hidden_size, num_layers, num_heads, batch_size, warmup_iters, num_iterations, seed, and vocab_size (32000, as 226/426) come from yaml. The collector runs two passes plus a summary row. output_format: csv Sweep dimensions: device_id, dtype, sequence_len, hidden_size, num_layers, num_heads, batch_size, warmup_iters.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| device_id | `--device-id` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| dtype | `--dtype` | smoke=FP16, baseline=BF16, extended=FP32 | BF16 | From Parameter list; see Execution Description With Parameters. |
| sequence_len | `--sequence-len` | smoke=16, baseline=128, extended=128 | 128 | From Parameter list; see Execution Description With Parameters. |
| hidden_size | `--hidden-size` | smoke=64, baseline=512, extended=512 | 512 | From Parameter list; see Execution Description With Parameters. |
| num_layers | `--num-layers` | smoke=1, baseline=4, extended=4 | 4 | From Parameter list; see Execution Description With Parameters. |
| num_heads | `--num-heads` | smoke=4, baseline=8, extended=8 | 8 | From Parameter list; see Execution Description With Parameters. |
| batch_size | `--batch-size` | smoke=1, baseline=8, extended=8 | 8 | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=1, baseline=3, extended=5 | 3 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=3, baseline=2630000, extended=7180000 | 2630000 | From Parameter list; see Execution Description With Parameters. |
| seed | `--seed` | smoke=42, baseline=42, extended=42 | 42 | From Parameter list; see Execution Description With Parameters. |
| vocab_size | `--vocab-size` | smoke=32000, baseline=32000, extended=32000 | 32000 | From Parameter list; see Execution Description With Parameters. |

## Invocation

```bash
Run gpu-bench-jax-xla-forwardpass.py via collect_jax_xla.py
```

## Raw Output Format

raw_results.csv with pass1, pass2, and summary, plus the two JAX transcripts in raw_output.txt

check,dtype,sequence_len,hidden_size,tflops_method,requested_iterations,completed_iterations,status,forward_latency_msec,tokens_per_sec,tflops,jit_compile_msec,peak_hbm_gb
pass1,bfloat16,128,256,modeled_from_transformer_operation_count,8,8,ok,4.0,32000,0.5,900,2.0

## Metrics

- **#1: Forward-pass latency, ms** — stored as `forward_latency_msec`.
- **#2: Throughput, tokens/s** — stored as `tokens_per_sec`.
- **#3: Modeled TFLOPS, full forward pass** — stored as `tflops`.
- **#4: XLA compilation time, ms** — stored as `jit_compile_msec`.
- **#5: Peak HBM, GB** — stored as `peak_hbm_gb`.

## Framework

Runs gpu-bench-jax-xla-forwardpass.py (the same script as 226/426): a small JAX/XLA transformer forward pass on synthetic tokens. dtype, sequence_len, hidden_size, num_layers, num_heads, batch_size, warmup_iters, num_iterations, seed, and vocab_size (32000, as 226/426) come from yaml. The collector runs two passes plus a summary row.

## Installation and Execution Summary

Run gpu-bench-jax-xla-forwardpass.py twice for yaml sequence_len, hidden_size, num_layers, num_heads, dtype, and batch_size, and parse RESULT forward_latency_msec, tokens_per_sec, tflops, jit_compile_msec, and peak_hbm_gb, to measure JAX/XLA transformer forward-pass performance

## Platform Portability

- **AMD (primary):** ```bash
Run gpu-bench-jax-xla-forwardpass.py via collect_jax_xla.py
```
- **NVIDIA:** Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

raw_results.csv with pass1, pass2, and summary, plus the two JAX transcripts in raw_output.txt

check,dtype,sequence_len,hidden_size,tflops_method,requested_iterations,completed_iterations,status,forward_latency_msec,tokens_per_sec,tflops,jit_compile_msec,peak_hbm_gb
pass1,bfloat16,128,256,modeled_from_transformer_operation_count,8,8,ok,4.0,32000,0.5,900,2.0

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Runs gpu-bench-jax-xla-forwardpass.py (the same script as 226/426): a small JAX/XLA transformer forward pass on synthetic tokens. dtype, sequence_len, hidden_size, num_layers, num_heads, batch_size, warmup_iters, num_iterations, seed, and vocab_size (32000, as 226/426) come from yaml. The collector runs two passes plus a summary row.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Runs gpu-bench-jax-xla-forwardpass.py (the same script as 226/426): a small JAX/XLA transformer forward pass on synthetic tokens. dtype, sequence_len, hidden_size, num_layers, num_heads, batch_size, warmup_iters, num_iterations, seed, and vocab_size (32000, as 226/426) come from yaml. The collector runs two passes plus a summary row.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
