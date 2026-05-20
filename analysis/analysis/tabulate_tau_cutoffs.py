"""
Purpose: Generate LaTeX table of equilibrium run-region cutoffs (tau-bar and
         tau-under) by treatment and seller position, for the theta values
         requested in the paper (theta = 1, 0.5; CRRA alpha = 1 - theta).
         tau-bar = highest belief at which selling begins (sigma > 0).
         tau-under = highest belief at which selling is certain (sigma = 1).
         Beliefs are reported as pi = Pr(z = G), consistent with the paper.
Author: Claude
Date: 2026-05-20
"""

from pathlib import Path

import numpy as np
import pandas as pd

# FILE PATHS
INPUT_PARQUET = Path("datastore/derived/equilibrium_sigma_grid.parquet")
OUTPUT_TEX = Path("analysis/output/tables/equilibrium_tau_cutoffs.tex")

# Parameterization: u = w^theta, so CRRA alpha = 1 - theta
THETAS = [1.0, 0.5]
SIGMA_TOL = 1e-6
# n holders -> seller position label (n = 1 never sells, omit)
POSITION_LABELS = {4: "1st", 3: "2nd", 2: "3rd"}
TREATMENTS = ["random", "average"]


# =====
# Main function
# =====
def main():
    """Read sigma grid, compute cutoffs, write LaTeX table."""
    grid = pd.read_parquet(INPUT_PARQUET)
    rows = build_cutoff_rows(grid)
    latex = build_latex_table(rows)
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(latex)
    print(f"Wrote {OUTPUT_TEX}")


# =====
# Cutoff extraction
# =====
def extract_cutoffs(sub, sigma_tol=SIGMA_TOL):
    """Return (tau_bar, tau_under) for one (treatment, alpha, n) slice.

    tau_bar: sup{pi : sigma > 0} (selling begins below this belief).
    tau_under: sup{pi : sigma = 1} (selling is certain below this belief).
    Returns NaN where the relevant region is empty.
    """
    selling = sub.loc[sub["sigma"] > sigma_tol, "pi"]
    full_run = sub.loc[sub["sigma"] >= 1 - sigma_tol, "pi"]
    tau_bar = selling.max() if len(selling) else np.nan
    tau_under = full_run.max() if len(full_run) else np.nan
    return tau_bar, tau_under


def nearest_alpha(grid, target_alpha):
    """Snap a target alpha to the closest value present in the grid."""
    available = np.array(sorted(grid["alpha"].unique()))
    return float(available[np.argmin(np.abs(available - target_alpha))])


def build_cutoff_rows(grid):
    """Build one record per (theta, seller position) with both treatments."""
    grid = grid[grid["n"] > 1]
    records = []
    for theta in THETAS:
        alpha = nearest_alpha(grid, 1.0 - theta)
        for n, position in POSITION_LABELS.items():
            record = {"theta": theta, "position": position}
            for treatment in TREATMENTS:
                mask = ((grid["treatment"] == treatment)
                        & (np.isclose(grid["alpha"], alpha))
                        & (grid["n"] == n))
                bar, under = extract_cutoffs(grid.loc[mask])
                record[f"{treatment}_bar"] = bar
                record[f"{treatment}_under"] = under
            records.append(record)
    return records


# =====
# LaTeX construction
# =====
def build_latex_table(rows):
    """Assemble the full LaTeX table string from cutoff records."""
    lines = _table_header()
    last_theta = None
    for row in rows:
        if row["theta"] != last_theta:
            lines.append(r"\midrule")
            last_theta = row["theta"]
        lines.append(_format_row(row))
    lines += _table_footer()
    return "\n".join(lines) + "\n"


def _table_header():
    """Return LaTeX header with treatment column groups."""
    return [
        r"\begingroup",
        r"\centering",
        r"\small",
        r"\begin{tabular}{cc|cc|cc}",
        r"\toprule",
        (r" & & \multicolumn{2}{c|}{Treatment 1 (Random)} "
         r"& \multicolumn{2}{c}{Treatment 2 (Average)} \\"),
        (r"$\theta$ & Seller & $\overline{\tau}$ & $\underline{\tau}$ "
         r"& $\overline{\tau}$ & $\underline{\tau}$ \\"),
    ]


def _format_row(row):
    """Format one (theta, position) record as a LaTeX row."""
    cells = [
        _format_theta(row["theta"]),
        row["position"],
        _fmt(row["random_bar"]),
        _fmt(row["random_under"]),
        _fmt(row["average_bar"]),
        _fmt(row["average_under"]),
    ]
    return " & ".join(cells) + r" \\"


def _format_theta(theta):
    """Format theta, dropping a trailing zero for whole numbers."""
    return f"{theta:g}"


def _fmt(value):
    """Format a belief cutoff for the table."""
    return "---" if pd.isna(value) else f"{value:.3f}"


def _table_footer():
    """Return LaTeX footer with explanatory note."""
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par",
        r"\vspace{0.5em}",
        r"\footnotesize",
        (r"\textit{Note:} Cutoffs are reported as $\pi = \Pr(z = G)$. "
         r"$\overline{\tau}$ is the highest belief at which selling begins "
         r"in equilibrium ($\sigma > 0$); $\underline{\tau}$ is the highest "
         r"belief at which selling is certain ($\sigma = 1$). A run occurs "
         r"for $\pi < \overline{\tau}$. Utility is $u = w^{\theta}$, so the "
         r"CRRA coefficient is $\alpha = 1 - \theta$ ($\theta = 1$ is risk "
         r"neutral). Values are read from the equilibrium $\sigma(n, \pi)$ "
         r"grid on the reachable-belief lattice."),
        r"\endgroup",
    ]


# %%
if __name__ == "__main__":
    main()
