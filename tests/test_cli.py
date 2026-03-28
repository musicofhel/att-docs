"""Tests for att.cli — command-line interface."""

import os
import tempfile
import subprocess
import sys



class TestCLI:
    def test_help_text(self):
        result = subprocess.run(
            [sys.executable, "-c", "from att.cli import main; main()"],
            capture_output=True, text=True,
            env={**os.environ, "COLUMNS": "80"},
        )
        # No subcommand → prints help and exits 0
        assert result.returncode == 0

    def test_benchmark_run_produces_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.yaml")
            output_path = os.path.join(tmpdir, "results.csv")

            # Write minimal config
            with open(config_path, "w") as f:
                f.write("""
system: coupled_lorenz
n_steps: 3000
coupling_values: [0.0, 0.5]
methods: [transfer_entropy, crqa]
normalization: rank
seed: 42
transient_discard: 500
""")

            result = subprocess.run(
                [sys.executable, "-c", "import sys; sys.argv[0] = 'att'; from att.cli import main; sys.exit(main())",
                 "benchmark", "run",
                 "--config", config_path,
                 "--output", output_path],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_path)

            # Verify CSV content
            import pandas as pd
            df = pd.read_csv(output_path)
            assert "coupling" in df.columns
            assert "method" in df.columns
            assert len(df) == 4  # 2 couplings * 2 methods

    def test_benchmark_run_with_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.yaml")
            output_path = os.path.join(tmpdir, "results.csv")
            plot_path = os.path.join(tmpdir, "results.png")

            with open(config_path, "w") as f:
                f.write("""
system: coupled_lorenz
n_steps: 3000
coupling_values: [0.0, 0.5]
methods: [transfer_entropy]
normalization: none
seed: 42
transient_discard: 500
""")

            result = subprocess.run(
                [sys.executable, "-c", "import sys; sys.argv[0] = 'att'; from att.cli import main; sys.exit(main())",
                 "benchmark", "run",
                 "--config", config_path,
                 "--output", output_path,
                 "--plot", plot_path],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(plot_path)

    def test_missing_config_errors(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.argv[0] = 'att'; from att.cli import main; sys.exit(main())",
             "benchmark", "run",
             "--config", "/nonexistent.yaml",
             "--output", "/tmp/out.csv"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
