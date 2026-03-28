"""ATT command-line interface."""

import argparse
import sys


def main():
    """Entry point for the `att` CLI."""
    parser = argparse.ArgumentParser(
        prog="att",
        description="Attractor Topology Toolkit — topological analysis of dynamical attractors",
    )
    subparsers = parser.add_subparsers(dest="command")

    # att benchmark run
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark coupling methods")
    bench_sub = bench_parser.add_subparsers(dest="bench_command")

    run_parser = bench_sub.add_parser("run", help="Run a coupling benchmark sweep")
    run_parser.add_argument("--config", required=True, help="YAML config file")
    run_parser.add_argument("--output", required=True, help="Output CSV file")
    run_parser.add_argument("--plot", default=None, help="Output plot PNG file")

    args = parser.parse_args()

    if args.command == "benchmark" and getattr(args, "bench_command", None) == "run":
        return _benchmark_run(args)

    parser.print_help()
    return 0


def _benchmark_run(args):
    """Execute a benchmark sweep from YAML config."""
    from pathlib import Path

    import numpy as np

    from att.config import load_config, set_seed
    from att.benchmarks import CouplingBenchmark
    from att.synthetic import generators

    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Resolve system generator
    system_name = config.get("system", "coupled_lorenz")
    system_fn = getattr(generators, system_name, None)
    if system_fn is None:
        print(f"Error: unknown system '{system_name}'", file=sys.stderr)
        return 1

    n_steps = config.get("n_steps", 10000)
    dt = config.get("dt", 0.01)

    def generator_fn(coupling, seed):
        return system_fn(n_steps=n_steps, dt=dt, coupling=coupling, seed=seed)

    coupling_values = config.get("coupling_values", list(np.linspace(0, 1, 11)))
    methods = config.get("methods", None)
    normalization = config.get("normalization", "rank")
    transient_discard = config.get("transient_discard", 1000)

    bench = CouplingBenchmark(methods=methods, normalization=normalization)
    df = bench.sweep(
        generator_fn=generator_fn,
        coupling_values=coupling_values,
        seed=seed,
        transient_discard=transient_discard,
    )

    # Save CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    # Optional plot
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from att.viz import plot_benchmark_sweep

        fig = plot_benchmark_sweep(df)
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
