"""
Purpose: Run our Python equilibrium solver and Munro's R solver on the same
         parameter set, then compare value functions and strategies numerically.
         Saves a comparison CSV and a diagnostic overlay plot.
Author: replication-engineer
Date: 2026-04-16
"""

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Archived under analysis/_archive/issue_109/analysis/. Make the project root
# importable so we can use the canonical package path for equilibrium_model.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.analysis import equilibrium_model as em  # noqa: E402

# FILE PATHS
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
MUNRO_RA_SCRIPT = ARCHIVE_ROOT / "analysis" / "munro_code" / "MixedStrat_low_RA.R"
R_DRIVER = ARCHIVE_ROOT / "analysis" / "_munro_driver.R"
OUTPUT_DIR = ARCHIVE_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
MUNRO_CSV_TMP = Path("/tmp/munro_replication_out.csv")
OUTPUT_CSV = OUTPUT_DIR / "munro_replication_comparison.csv"
OUTPUT_PNG = PLOTS_DIR / "munro_replication_comparison.png"

# Matched parameters (Munro's RA file: low-liquidity, risk-averse, CRRA a=0.5)
ALPHA = 0.5
TREATMENT = "random"  # Munro averages u(prices), matching our "random" treatment
# Reversed from Munro's descending c(8, 5.5, 3, 0.5): our equilibrium_model
# assumes PRICES is ASCENDING (rho() uses PRICES[n-1-j] for the first seller,
# forced_liquidation() uses PRICES[:m] for the lowest-price tail).
MUNRO_PRICES = [0.5, 3.0, 5.5, 8.0]


# =====
# Main function
# =====
def main():
    """Run both solvers, build comparison, save CSV + PNG."""
    _require_rscript()
    print(f"[1/4] Running Munro R solver: {MUNRO_RA_SCRIPT.name}")
    munro_df = run_munro_r_solver()
    print(f"[2/4] Running our Python solver (alpha={ALPHA}, "
          f"treatment={TREATMENT}) on Munro's price schedule")
    # Munro's `belief` column is P(Bad); our solver is indexed by pi = P(Good).
    # Sample our outputs at 1 - munro_beliefs so both are aligned on the same state.
    munro_p_bad = munro_df["belief"].to_numpy()
    ours = run_our_solver_on_munro_params(1.0 - munro_p_bad)
    print("[3/4] Building numerical comparison")
    comp_df = build_comparison(munro_df, ours)
    _write_outputs(comp_df, ours)
    print("[4/4] Summary of diffs (our_sigma - munro_sigma):")
    _print_diff_summary(comp_df)
    print(f"Wrote CSV : {OUTPUT_CSV}")
    print(f"Wrote PNG : {OUTPUT_PNG}")


# =====
# Environment checks
# =====
def _require_rscript():
    """Fail loudly if Rscript is not on PATH."""
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R (https://cran.r-project.org) "
            "or add Rscript to PATH, then re-run this script.")
    if not MUNRO_RA_SCRIPT.exists():
        raise FileNotFoundError(
            f"Munro R script missing: {MUNRO_RA_SCRIPT}. Check analysis/analysis/munro_code/.")
    if not R_DRIVER.exists():
        raise FileNotFoundError(f"R driver missing: {R_DRIVER}.")


# =====
# Munro R solver
# =====
def run_munro_r_solver():
    """Invoke the R driver and load its CSV output into a DataFrame."""
    MUNRO_CSV_TMP.unlink(missing_ok=True)
    cmd = ["Rscript", "--vanilla", str(R_DRIVER),
           str(MUNRO_RA_SCRIPT), str(MUNRO_CSV_TMP)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Rscript failed (code {result.returncode}).\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}")
    if not MUNRO_CSV_TMP.exists():
        raise FileNotFoundError(
            f"R driver did not write {MUNRO_CSV_TMP}. STDOUT:\n{result.stdout}")
    return pd.read_csv(MUNRO_CSV_TMP)


# =====
# Our Python solver configured to Munro's primitives
# =====
def run_our_solver_on_munro_params(target_p_good):
    """Solve our Bellman on Munro's price schedule, sample at pi = P(Good) values."""
    _apply_munro_params_to_module()
    result = em.solve_equilibrium(alpha=ALPHA, treatment=TREATMENT,
                                  t_max=20, tol=1e-8, max_iter=1000)
    grid = result["belief_grid"]
    sigma = {n: np.interp(target_p_good, grid, result["sigma"][n])
             for n in range(1, em.N_INVESTORS + 1)}
    v_vals = {n: np.interp(target_p_good, grid, result["v_table"][n])
              for n in range(1, em.N_INVESTORS + 1)}
    thresholds = {n: em.find_continuous_threshold(
        n, ALPHA, TREATMENT, result["v_table"], grid)
        for n in range(1, em.N_INVESTORS + 1)}
    return {"belief_p_good": target_p_good, "sigma": sigma,
            "v_table": v_vals, "thresholds": thresholds, "native_grid": grid,
            "native_sigma": result["sigma"], "native_v": result["v_table"]}


def _apply_munro_params_to_module():
    """Override our module-level constants with Munro's values at runtime."""
    em.PRICES = list(MUNRO_PRICES)
    em.FINAL_VALUE = 20.0
    em.MU_B = 0.675
    em.MU_G = 1.0 - 0.675
    em.LAMBDA = 0.125
    em.PI_0 = 0.5


# =====
# Comparison construction
# =====
def build_comparison(munro_df, ours):
    """Assemble a tidy DataFrame with both methods' sigma / V per belief and n.

    Aligned on the same state: Munro's belief is P(Bad); our pi = P(Good) = 1 - P(Bad).
    """
    rows = []
    munro_p_bad = munro_df["belief"].to_numpy()
    for n in range(1, em.N_INVESTORS + 1):
        for i, p_bad in enumerate(munro_p_bad):
            rows.append(_comparison_row(n, i, p_bad, munro_df, ours))
    return pd.DataFrame(rows)


def _comparison_row(n, i, p_bad, munro_df, ours):
    """Build one row of the comparison table for investor count n at grid index i."""
    munro_sig = _munro_sigma_at(munro_df, n, i)
    our_sig = float(ours["sigma"][n][i])
    munro_v = float(munro_df[f"V_{n}"].to_numpy()[i])
    our_v = float(ours["v_table"][n][i])
    return {
        "n": n, "belief_p_bad": p_bad, "belief_p_good": 1.0 - p_bad,
        "munro_sigma": munro_sig, "our_sigma": our_sig,
        "sigma_diff": our_sig - munro_sig,
        "munro_V": munro_v, "our_V": our_v, "V_diff": our_v - munro_v,
    }


def _munro_sigma_at(df, n, i):
    """Look up Munro's sigma at a grid index (n=1 has no sigma column)."""
    if n == 1:
        return 0.0
    return float(df[f"sigma_n{n}"].to_numpy()[i])


# =====
# Reporting and output
# =====
def _print_diff_summary(comp_df):
    """Per-n summary of strategy and value differences."""
    for n in range(1, em.N_INVESTORS + 1):
        sub = comp_df[comp_df["n"] == n]
        sig_max = sub["sigma_diff"].abs().max()
        sig_mean = sub["sigma_diff"].abs().mean()
        v_max = sub["V_diff"].abs().max()
        v_mean = sub["V_diff"].abs().mean()
        corr = _safe_corr(sub["our_sigma"], sub["munro_sigma"])
        print(f"  n={n}: |dsigma| max={sig_max:.4f} mean={sig_mean:.4f} "
              f"corr={corr:.4f} | |dV| max={v_max:.4f} mean={v_mean:.4f}")


def _safe_corr(a, b):
    """Correlation that returns NaN safely when a column is constant."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _write_outputs(comp_df, ours):
    """Persist CSV and diagnostic plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(OUTPUT_CSV, index=False)
    _save_overlay_plot(comp_df, ours)


def _save_overlay_plot(comp_df, ours):
    """Overlay our sigma vs. Munro's across beliefs for n in {2, 3, 4}."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, n in zip(axes, (2, 3, 4)):
        sub = comp_df[comp_df["n"] == n].sort_values("belief_p_good")
        ax.plot(sub["belief_p_good"], sub["munro_sigma"], "o-", label="Munro (R)")
        ax.plot(sub["belief_p_good"], sub["our_sigma"], "s--", label="Ours (Py)")
        ax.axvline(ours["thresholds"][n], color="grey", ls=":",
                   label=f"our pi*={ours['thresholds'][n]:.3f}")
        ax.set_xlabel("belief pi = P(Good)  (Munro P(Bad) flipped to 1 - P(Bad))")
        ax.set_title(f"n = {n}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("sigma (sell probability)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)


# %%
if __name__ == "__main__":
    sys.exit(main())
