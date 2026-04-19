"""
Purpose: Cross-validate our Python Bellman solver against Munro's R V-iteration
         style solver on OUR native price schedule PRICES=[2,4,6,8]. Symmetric
         to compare_munro_replication.py, which runs ours on Munro's prices.
Author: comparison-engineer
Date: 2026-04-16
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Archived under analysis/_archive/issue_109/analysis/. Re-point sys.path so
# the canonical package import resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.analysis import equilibrium_model as em  # noqa: E402

# FILE PATHS
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ARCHIVE_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
MUNRO_STYLE_CSV = OUTPUT_DIR / "munro_style_our_prices.csv"
OUTPUT_CSV = OUTPUT_DIR / "cross_validation_our_prices.csv"
OUTPUT_PNG = PLOTS_DIR / "cross_validation_our_prices.png"

# Native-config parameters: our production PRICES=[2,4,6,8], alpha=0.5, random.
ALPHA = 0.5
TREATMENT = "random"
T_MAX = 20


# =====
# Main function
# =====
def main():
    """Run our solver on native prices, compare against Munro-style R output."""
    print(f"[1/4] Loading Munro-style R output: {MUNRO_STYLE_CSV.name}")
    munro_df = _load_munro_style_csv()
    print(f"[2/4] Running our Python solver (alpha={ALPHA}, "
          f"treatment={TREATMENT}, PRICES={em.PRICES})")
    munro_p_bad = np.sort(munro_df["belief"].unique())
    ours = run_our_solver(1.0 - munro_p_bad)
    print("[3/4] Building numerical comparison")
    comp_df = build_comparison(munro_df, ours, munro_p_bad)
    _write_outputs(comp_df)
    print("[4/4] Summary of diffs (our_sigma - munro_sigma):")
    _print_diff_summary(comp_df)
    print(f"Wrote CSV : {OUTPUT_CSV}")
    print(f"Wrote PNG : {OUTPUT_PNG}")


# =====
# Input loading
# =====
def _load_munro_style_csv():
    """Load Munro-style R solver CSV; fail loudly if task #2 hasn't run."""
    if not MUNRO_STYLE_CSV.exists():
        raise FileNotFoundError(
            f"Munro-style R output missing: {MUNRO_STYLE_CSV}. "
            f"Run the task #2 R driver (_munro_style_solver.R) first to "
            f"generate this file with our PRICES=[2,4,6,8].")
    df = pd.read_csv(MUNRO_STYLE_CSV)
    required = {"n", "belief", "sigma", "V"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{MUNRO_STYLE_CSV} missing required columns: {sorted(missing)}. "
            f"Expected {sorted(required)}, got {sorted(df.columns)}.")
    return df


# =====
# Our Python solver (native PRICES, no monkey-patching)
# =====
def run_our_solver(target_p_good):
    """Solve our Bellman with native PRICES, sample at pi = P(Good) values."""
    try:
        result = em.solve_equilibrium(alpha=ALPHA, treatment=TREATMENT,
                                      t_max=T_MAX, tol=1e-8, max_iter=1000)
    except Exception as exc:
        raise RuntimeError(
            f"Our solver failed (alpha={ALPHA}, treatment={TREATMENT}, "
            f"t_max={T_MAX}): {exc}") from exc
    grid = result["belief_grid"]
    sigma = {n: np.interp(target_p_good, grid, result["sigma"][n])
             for n in range(1, em.N_INVESTORS + 1)}
    v_vals = {n: np.interp(target_p_good, grid, result["v_table"][n])
              for n in range(1, em.N_INVESTORS + 1)}
    return {"belief_p_good": target_p_good,
            "sigma": sigma, "v_table": v_vals}


# =====
# Comparison construction
# =====
def build_comparison(munro_df, ours, munro_p_bad):
    """Tidy DataFrame with both methods' sigma / V per belief and n.

    Munro's `belief` is P(Bad); our pi = P(Good) = 1 - P(Bad).
    """
    rows = []
    for n in range(1, em.N_INVESTORS + 1):
        sub = munro_df[munro_df["n"] == n].sort_values("belief").reset_index(drop=True)
        _require_belief_alignment(sub, munro_p_bad, n)
        for i, p_bad in enumerate(munro_p_bad):
            rows.append(_comparison_row(n, i, p_bad, sub, ours))
    return pd.DataFrame(rows)


def _require_belief_alignment(sub, munro_p_bad, n):
    """Fail loudly if Munro belief grid for n doesn't match the combined grid."""
    got = sub["belief"].to_numpy()
    if len(got) != len(munro_p_bad) or not np.allclose(got, munro_p_bad):
        raise ValueError(
            f"Belief grid mismatch at n={n}: R CSV has {len(got)} points "
            f"vs expected {len(munro_p_bad)}. Re-run the R solver.")


def _comparison_row(n, i, p_bad, munro_sub, ours):
    """Build one comparison row for investor count n at grid index i."""
    munro_sig = float(munro_sub["sigma"].to_numpy()[i])
    munro_v = float(munro_sub["V"].to_numpy()[i])
    our_sig = float(ours["sigma"][n][i])
    our_v = float(ours["v_table"][n][i])
    return {
        "n": n, "belief_p_bad": p_bad, "belief_p_good": 1.0 - p_bad,
        "munro_sigma": munro_sig, "our_sigma": our_sig,
        "sigma_diff": our_sig - munro_sig,
        "munro_V": munro_v, "our_V": our_v, "V_diff": our_v - munro_v,
    }


# =====
# Reporting and output
# =====
def _print_diff_summary(comp_df):
    """Per-n summary of strategy and value differences with worst-case belief."""
    for n in range(1, em.N_INVESTORS + 1):
        sub = comp_df[comp_df["n"] == n]
        sig_max = sub["sigma_diff"].abs().max()
        sig_mean = sub["sigma_diff"].abs().mean()
        v_max = sub["V_diff"].abs().max()
        v_mean = sub["V_diff"].abs().mean()
        corr = _safe_corr(sub["our_sigma"], sub["munro_sigma"])
        sig_at = float(sub.loc[sub["sigma_diff"].abs().idxmax(), "belief_p_bad"])
        v_at = float(sub.loc[sub["V_diff"].abs().idxmax(), "belief_p_bad"])
        print(f"  n={n}: |dsigma| max={sig_max:.4f} mean={sig_mean:.4f} "
              f"corr={corr:.4f} | |dV| max={v_max:.4f} mean={v_mean:.4f} "
              f"| worst sigma@p_bad={sig_at:.4f} V@p_bad={v_at:.4f}")


def _safe_corr(a, b):
    """Pearson correlation; NaN when either series is constant."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _write_outputs(comp_df):
    """Persist CSV and diagnostic overlay plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(OUTPUT_CSV, index=False)
    _save_overlay_plot(comp_df)


def _save_overlay_plot(comp_df):
    """Overlay our sigma vs Munro-style sigma across beliefs, per-n panels."""
    fig, axes = plt.subplots(1, em.N_INVESTORS, figsize=(16, 4), sharey=True)
    for ax, n in zip(axes, range(1, em.N_INVESTORS + 1)):
        sub = comp_df[comp_df["n"] == n].sort_values("belief_p_good")
        ax.plot(sub["belief_p_good"], sub["munro_sigma"], "o-", label="Munro-style (R)")
        ax.plot(sub["belief_p_good"], sub["our_sigma"], "s--", label="Ours (Py)")
        ax.set_xlabel("belief pi = P(Good)")
        ax.set_title(f"n = {n}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("sigma (sell probability)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)


# %%
if __name__ == "__main__":
    sys.exit(main())
