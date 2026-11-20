# Branch comparison benchmarks

Run the same deterministic planner-level cases on each branch:

```powershell
python examples/benchmark_synthetic.py --iterations 50
```

The script warms the representative code paths, then reports steady-state p50/p95/max latency, validity, stop reasons, and cycles over the 30 ms deadline. Use the same Python/dependency environment for a fair branch comparison. Keep EUFS/FSDS lap-time and cone-contact results separate; these 2D fixtures are useful algorithm checks, not simulator or vehicle validation.

The current local run is recorded in [v0.1 baseline](baseline-v0.1.md).
