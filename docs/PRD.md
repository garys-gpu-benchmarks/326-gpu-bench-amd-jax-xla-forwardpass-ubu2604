# PRD.md:  "The Why"; Product requirements, benchmark metadata table, high-level requirements, etc.

Product Requirements Document

"The Why"; Product requirements, benchmark metadata table, high-level requirements, etc. Defines the benchmark goal, validation objective, test name, benchmark number, category, and high-level success criteria.

## Benchmark Matrix Document Metadata (via benchmark_specification.json)

This PRD.md section is populated from benchmark_specification.json, which is the structured source of benchmark-specific product requirements.

## Workload Number
326

## Workload Name
JAX XLA Transformer Forward Pass

## Execution Summary (Run and Measure)
Run gpu-bench-jax-xla-forwardpass.py twice for yaml sequence_len, hidden_size, num_layers, num_heads, dtype, and batch_size, and parse RESULT forward_latency_msec, tokens_per_sec, tflops, jit_compile_msec, and peak_hbm_gb, to measure JAX/XLA transformer forward-pass performance

## Main Goal
Measure JAX/XLA transformer forward-pass performance

## Validation Objective
Validates JAX JIT compile time and forward-pass latency, throughput, and TFLOPS on the synthetic transformer

## Workload Category
Training, Inference, Model Workloads

## Validation Requirement

The benchmark must include an automated SQLite-integrated validation layer that verifies persisted results from `results/benchmark.db`. Validation must confirm:

1. The benchmark run completed successfully with no tool errors.
2. Required samples and aggregate metrics were persisted for every swept shape.
3. Metrics are finite and physically sensible (positive, within plausible bounds).
4. Measured values satisfy configured thresholds when the workload defines pass/fail gates.
5. The benchmark fails validation when required data is missing, invalid, or outside bounds.

## Non-Functional Requirements

| Requirement | Target |
|---|---|
| Automation | Runs to completion without manual intervention after `bash run_benchmark.sh` |
| Idempotency | Re-running `run_benchmark.sh` appends a new run; never corrupts existing rows |
| Persistence | All metrics survive script exit; `results/benchmark.db` is the durable record |
