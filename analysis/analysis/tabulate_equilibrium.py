"""
Purpose: Generate LaTeX table of equilibrium predictions — average P(Bad)
         at each seller position, matching M&M (2020) Table 2 format.
Author: Claude
Date: 2026-04-06
"""

from pathlib import Path

import pandas as pd

# FILE PATHS
INPUT_CSV = Path("datastore/derived/equilibrium_thresholds.csv")
OUTPUT_TEX = Path("analysis/output/tables/equilibrium_thresholds.tex")


# =====
# Main function
# =====
def main():
    """Read simulation data and generate LaTeX table."""
    df = pd.read_csv(INPUT_CSV)
    df = df[df["n"] > 1]  # n=1 never sells, omit from table
    # Convert avg_pi_at_sale (P(Good)) to P(Bad) for M&M convention
    df["avg_pbad_at_sale"] = 1 - df["avg_pi_at_sale"]
    latex = build_latex_table(df)
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(latex)
    print(f"Wrote {OUTPUT_TEX}")


# =====
# Table construction
# =====
def build_latex_table(df):
    """Build complete booktabs LaTeX table string."""
    lines = _table_header()
    alphas = sorted(df["alpha"].unique())
    for alpha in alphas:
        lines.append(_format_row(df, alpha))
    lines += _table_footer()
    return "\n".join(lines) + "\n"


def _table_header():
    """Return LaTeX table header: seller position x treatment."""
    return [
        r"\begingroup",
        r"\centering",
        r"\small",
        r"\begin{tabular}{c|ccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{Random} & \multicolumn{3}{c}{Average} \\",
        (r"$\alpha$ & 1st seller & 2nd seller & 3rd seller "
         r"& 1st seller & 2nd seller & 3rd seller \\"),
        r"\midrule",
    ]


def _format_row(df, alpha):
    """Format one alpha value as a LaTeX table row."""
    cells = [_format_alpha(alpha)]
    for treatment in ["random", "average"]:
        # n=4 → 1st seller, n=3 → 2nd, n=2 → 3rd
        for n in [4, 3, 2]:
            val = _get_avg_pbad(df, alpha, treatment, n)
            cells.append(_format_value(val))
    return " & ".join(cells) + r" \\"


def _format_alpha(alpha):
    """Format alpha value for display."""
    if alpha == 0.0:
        return "0"
    return f"{alpha:.1f}"


def _get_avg_pbad(df, alpha, treatment, n):
    """Look up average P(Bad) at sale for given parameters."""
    mask = ((df["alpha"] == alpha) & (df["treatment"] == treatment)
            & (df["n"] == n))
    return df.loc[mask, "avg_pbad_at_sale"].values[0]


def _format_value(val):
    """Format a P(Bad) value for the table."""
    if pd.isna(val):
        return "---"
    return f"{val:.3f}"


def _table_footer():
    """Return LaTeX table footer with notes."""
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par",
        r"\vspace{0.5em}",
        r"\footnotesize",
        (r"\textit{Note:} Each cell reports the average $\pi$ "
         r"(probability of bad state) at the time of the $k$th sale, "
         r"from 10{,}000 simulated games using equilibrium strategies."),
        (r"At $\alpha = 0$ (risk neutral), predictions are identical "
         r"across treatments."),
        r"The 4th investor never sells in equilibrium ($n=1$ boundary).",
        r"\endgroup",
    ]


# %%
if __name__ == "__main__":
    main()
