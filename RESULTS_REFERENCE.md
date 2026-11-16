# Results: 326 JAX XLA Transformer Forward Pass

**Main goal:** Measure JAX/XLA transformer forward-pass performance  
**Tools:** Run gpu-bench-jax-xla-forwardpass.py via collect_jax_xla.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_143 · recorded 2026-11-16). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× AMD Instinct MI300X VF (196288 MB) · INTEL(R) XEON(R) PLATINUM 8568Y+ · Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic · driver 7.1.9.31600000
- **Run:** 2026-11-16T20:04:34Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:21 · benchmark 00:00 · whole run 01:30
- **Source:** run 326_20261116_200434
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype FP16 --sequence-len 16 --hidden-size 64 --num-layers 1 --num-heads 4 --vocab-size 32000 --batch-size 1 --warmup-iters 1 --num-iterations 3 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | Forward-pass latency, ms (forward_latency_msec) | 0.115613 |
| 2 | Throughput, tokens/s (tokens_per_sec) | 138392.342282 |
| 3 | Modeled TFLOPS, full forward pass (tflops) | 0.576492 |
| 4 | XLA compilation time, ms (jit_compile_msec) | 2811.581178 |
| 5 | Peak HBM, GB (peak_hbm_gb) | 0.188677 |
<!-- /profile:smoke -->
