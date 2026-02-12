"""
Purpose: Validate unified logit regression output tables against expected values
         and cross-check AME signs with LPM coefficients.
Author: Claude Code
Date: 2026-02-10

Tests verify that:
1. The logit script runs to completion and produces both output files
2. AME signs match LPM coefficient signs for jointly-significant variables
3. Observation counts match expected values from real output
4. The manual delta method for feglm AMEs matches marginaleffects::avg_slopes()
"""

import subprocess
import re
import pytest
from pathlib import Path

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGIT_SCRIPT = PROJECT_ROOT / "analysis" / "analysis" / "unified_selling_logit.R"
LOGIT_TABLE = PROJECT_ROOT / "analysis" / "output" / "tables" / "unified_selling_logit.tex"
LOGIT_TABLE_FULL = PROJECT_ROOT / "analysis" / "output" / "tables" / "unified_selling_logit_full.tex"
LPM_TABLE_FULL = PROJECT_ROOT / "analysis" / "output" / "tables" / "unified_selling_regression_full.tex"


# =====
# Main function
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def logit_full_text():
    """Load the full logit table LaTeX source."""
    if not LOGIT_TABLE_FULL.exists():
        pytest.skip(f"Logit full table not found: {LOGIT_TABLE_FULL}")
    return LOGIT_TABLE_FULL.read_text()


@pytest.fixture(scope="module")
def logit_compact_text():
    """Load the compact logit table LaTeX source."""
    if not LOGIT_TABLE.exists():
        pytest.skip(f"Logit compact table not found: {LOGIT_TABLE}")
    return LOGIT_TABLE.read_text()


@pytest.fixture(scope="module")
def lpm_full_text():
    """Load the full LPM table LaTeX source."""
    if not LPM_TABLE_FULL.exists():
        pytest.skip(f"LPM full table not found: {LPM_TABLE_FULL}")
    return LPM_TABLE_FULL.read_text()


@pytest.fixture(scope="module")
def logit_panels(logit_full_text):
    """Parse the logit full table into three panel dicts."""
    return parse_panels(logit_full_text)


@pytest.fixture(scope="module")
def lpm_panels(lpm_full_text):
    """Parse the LPM full table into three panel dicts."""
    return parse_panels(lpm_full_text)


# =====
# LaTeX table parsing helpers
# =====
def parse_panels(tex_text):
    """Split a unified table into Panel A, B, C coefficient dicts."""
    panel_labels = [
        "Panel A: All Participants",
        "Panel B: Second Sellers",
        "Panel C: First Sellers",
    ]
    panels = {}
    for i, label in enumerate(panel_labels):
        end_label = panel_labels[i + 1] if i + 1 < len(panel_labels) else None
        block = extract_panel_block(tex_text, label, end_label)
        panels[label] = parse_coef_block(block)
    return panels


def extract_panel_block(tex_text, start_label, end_label):
    """Extract text between panel header and next panel or end."""
    start_pat = re.escape(start_label)
    start_match = re.search(start_pat, tex_text)
    if not start_match:
        return ""
    start_pos = start_match.end()
    if end_label:
        end_match = re.search(re.escape(end_label), tex_text[start_pos:])
        end_pos = start_pos + end_match.start() if end_match else len(tex_text)
    else:
        end_pos = len(tex_text)
    return tex_text[start_pos:end_pos]


def parse_coef_block(block):
    """Parse a panel block into {(var_label, col): (estimate, has_stars)}."""
    coefs = {}
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        row = parse_coef_row(lines, i)
        if row is not None:
            label, values = row
            for col_idx, (est, stars) in enumerate(values):
                if est is not None:
                    coefs[(label, col_idx)] = (est, stars)
            i += 2
        else:
            i += 1
    return coefs


def parse_coef_row(lines, idx):
    """Try to parse a coefficient + SE row pair starting at idx."""
    if idx + 1 >= len(lines):
        return None
    line = lines[idx].strip()
    if not ("&" in line and "\\\\" in line):
        return None
    # Skip headers, fit stats, and section separators
    if any(kw in line for kw in ["emph{", "midrule", "Observations",
                                  "Pseudo", "Log-lik", "R$^2$",
                                  "Constant", "multicolumn"]):
        return None
    return extract_label_and_values(line)


def extract_label_and_values(line):
    """Extract variable label and (estimate, has_stars) from a coef line."""
    parts = re.split(r"&", line.replace("\\\\", ""))
    if len(parts) < 2:
        return None
    label = parts[0].strip()
    values = []
    for cell in parts[1:]:
        cell = cell.strip()
        if cell == "":
            values.append((None, False))
        else:
            est, stars = parse_estimate_cell(cell)
            values.append((est, stars))
    return label, values


def parse_estimate_cell(cell):
    """Extract numeric estimate and significance from a LaTeX cell."""
    has_stars = "$^{" in cell
    num_match = re.search(r"(-?\d+\.\d+)", cell)
    if num_match:
        return float(num_match.group(1)), has_stars
    return None, False


def parse_obs_from_panel(block):
    """Extract observation counts (list of 3) from a panel block."""
    for line in block.split("\n"):
        if "Observations" in line:
            nums = re.findall(r"([\d,]+)", line.replace("Observations", ""))
            return [int(n.replace(",", "")) for n in nums]
    return []


# =====
# Test 1: Model convergence (slow)
# =====
class TestModelConvergence:
    """Verify the logit script runs without errors."""

    @pytest.mark.slow
    def test_logit_script_completes(self):
        """Run unified_selling_logit.R and check exit code 0."""
        result = subprocess.run(
            ["Rscript", str(LOGIT_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,
        )
        assert result.returncode == 0, (
            f"Logit script failed with code {result.returncode}.\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    @pytest.mark.slow
    def test_logit_produces_compact_table(self):
        """Compact logit table file exists after script run."""
        assert LOGIT_TABLE.exists(), f"Missing: {LOGIT_TABLE}"

    @pytest.mark.slow
    def test_logit_produces_full_table(self):
        """Full logit table file exists after script run."""
        assert LOGIT_TABLE_FULL.exists(), f"Missing: {LOGIT_TABLE_FULL}"


# =====
# Test 2: AME signs match LPM coefficient signs
# =====
class TestAmeSignsMatchLpm:
    """For jointly-significant coefficients, logit AME and LPM agree on sign."""

    def test_panel_a_signs_match(self, logit_panels, lpm_panels):
        """Panel A: AME signs match LPM signs for significant variables."""
        mismatches = check_sign_agreement(
            logit_panels["Panel A: All Participants"],
            lpm_panels["Panel A: All Participants"],
        )
        assert len(mismatches) == 0, format_sign_mismatches(mismatches)

    def test_panel_b_signs_match(self, logit_panels, lpm_panels):
        """Panel B: AME signs match LPM signs for significant variables."""
        mismatches = check_sign_agreement(
            logit_panels["Panel B: Second Sellers"],
            lpm_panels["Panel B: Second Sellers"],
        )
        assert len(mismatches) == 0, format_sign_mismatches(mismatches)

    def test_panel_c_signs_match(self, logit_panels, lpm_panels):
        """Panel C: AME signs match LPM signs for significant variables."""
        mismatches = check_sign_agreement(
            logit_panels["Panel C: First Sellers"],
            lpm_panels["Panel C: First Sellers"],
        )
        assert len(mismatches) == 0, format_sign_mismatches(mismatches)


def check_sign_agreement(logit_coefs, lpm_coefs):
    """Return list of (key, logit_est, lpm_est) where signs disagree."""
    mismatches = []
    for key in logit_coefs:
        if key not in lpm_coefs:
            continue
        logit_est, logit_sig = logit_coefs[key]
        lpm_est, lpm_sig = lpm_coefs[key]
        if not (logit_sig and lpm_sig):
            continue
        if signs_disagree(logit_est, lpm_est):
            mismatches.append((key, logit_est, lpm_est))
    return mismatches


def signs_disagree(a, b):
    """True if a and b have opposite signs (both nonzero)."""
    if a == 0 or b == 0:
        return False
    return (a > 0) != (b > 0)


def format_sign_mismatches(mismatches):
    """Format sign mismatches for assertion message."""
    lines = ["Sign mismatches (both significant):"]
    for key, logit_est, lpm_est in mismatches:
        lines.append(f"  {key}: logit={logit_est}, lpm={lpm_est}")
    return "\n".join(lines)


# =====
# Test 3: Observation counts match expected values
# =====
class TestObservationCounts:
    """Verify observation counts from real output."""

    def test_panel_a_obs(self, logit_full_text):
        """Panel A: ~13,700 for M1/M3, ~12,300 for M2."""
        obs = get_panel_obs(logit_full_text, "Panel A: All Participants")
        assert_obs_in_range(obs[0], 13000, 14500, "Panel A M1")
        assert_obs_in_range(obs[1], 11500, 13000, "Panel A M2")
        assert_obs_in_range(obs[2], 13000, 14500, "Panel A M3")

    def test_panel_a_exact_obs(self, logit_full_text):
        """Panel A exact values from verified output."""
        obs = get_panel_obs(logit_full_text, "Panel A: All Participants")
        assert obs[0] == 13713, f"Panel A M1: {obs[0]} != 13713"
        assert obs[1] == 12369, f"Panel A M2: {obs[1]} != 12369"
        assert obs[2] == 13590, f"Panel A M3: {obs[2]} != 13590"

    def test_panel_b_obs(self, logit_full_text):
        """Panel B: ~620 obs."""
        obs = get_panel_obs(logit_full_text, "Panel B: Second Sellers")
        assert_obs_in_range(obs[0], 550, 700, "Panel B M1")
        assert_obs_in_range(obs[1], 550, 700, "Panel B M2")
        assert_obs_in_range(obs[2], 550, 700, "Panel B M3")

    def test_panel_b_exact_obs(self, logit_full_text):
        """Panel B exact values from verified output."""
        obs = get_panel_obs(logit_full_text, "Panel B: Second Sellers")
        assert obs[0] == 622, f"Panel B M1: {obs[0]} != 622"
        assert obs[1] == 622, f"Panel B M2: {obs[1]} != 622"
        assert obs[2] == 619, f"Panel B M3: {obs[2]} != 619"

    def test_panel_c_obs(self, logit_full_text):
        """Panel C: ~1,200 for M1, ~1,190 for M2."""
        obs = get_panel_obs(logit_full_text, "Panel C: First Sellers")
        assert_obs_in_range(obs[0], 1100, 1350, "Panel C M1")
        assert_obs_in_range(obs[1], 1100, 1350, "Panel C M2")
        assert_obs_in_range(obs[2], 1050, 1300, "Panel C M3")

    def test_panel_c_exact_obs(self, logit_full_text):
        """Panel C exact values from verified output."""
        obs = get_panel_obs(logit_full_text, "Panel C: First Sellers")
        assert obs[0] == 1218, f"Panel C M1: {obs[0]} != 1218"
        assert obs[1] == 1194, f"Panel C M2: {obs[1]} != 1194"
        assert obs[2] == 1183, f"Panel C M3: {obs[2]} != 1183"


def get_panel_obs(tex_text, panel_label):
    """Extract observation counts for a specific panel."""
    panels = parse_panels(tex_text)
    # Re-extract the raw block for obs parsing
    panel_labels = [
        "Panel A: All Participants",
        "Panel B: Second Sellers",
        "Panel C: First Sellers",
    ]
    idx = panel_labels.index(panel_label)
    end_label = panel_labels[idx + 1] if idx + 1 < len(panel_labels) else None
    block = extract_panel_block(tex_text, panel_label, end_label)
    return parse_obs_from_panel(block)


def assert_obs_in_range(obs, low, high, label):
    """Assert an observation count is within expected range."""
    assert low <= obs <= high, (
        f"{label}: obs={obs} not in [{low}, {high}]"
    )


# =====
# Test 4: Delta method AME sanity check
# =====
class TestDeltaMethodAme:
    """Validate manual delta method against marginaleffects::avg_slopes()."""

    def test_delta_method_matches_avg_slopes(self):
        """Manual delta method matches avg_slopes() on a simple logit."""
        r_script = _build_delta_method_r_script()
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"R script failed.\nSTDERR:\n{result.stderr[-2000:]}"
        )
        _verify_delta_method_output(result.stdout)


def _build_delta_method_r_script():
    """Build the R script that compares delta method vs avg_slopes."""
    return """
library(marginaleffects)

# Load and prepare data (same as logit script)
df <- read.csv("datastore/derived/emotions_traits_selling_dataset.csv")
df <- df[df$already_sold == 0, ]
df$segment <- as.factor(df$segment)

# Fit simple logit WITHOUT absorbed FE
model <- glm(sold ~ signal + period, family = binomial, data = df)

# Method 1: marginaleffects::avg_slopes()
ame_ref <- avg_slopes(model)
ame_ref_df <- as.data.frame(ame_ref)

# Method 2: Manual delta method (same logic as extract_ame_fixest)
betas <- coef(model)
phat <- predict(model, type = "response")
w <- phat * (1 - phat)
X <- model.matrix(model)
p <- length(betas)
ame_manual <- sapply(seq_len(p), function(j) mean(betas[j] * w))
names(ame_manual) <- names(betas)

# Compare for non-intercept terms
for (term_name in c("signal", "period")) {
  ref_row <- ame_ref_df[ame_ref_df$term == term_name, ]
  ref_val <- ref_row$estimate
  manual_val <- ame_manual[term_name]
  diff <- abs(ref_val - manual_val)
  cat(sprintf("TERM=%s REF=%.10f MANUAL=%.10f DIFF=%.2e\\n",
              term_name, ref_val, manual_val, diff))
}
cat("DELTA_METHOD_TEST_COMPLETE\\n")
"""


def _verify_delta_method_output(stdout):
    """Parse R output and check delta method matches avg_slopes."""
    assert "DELTA_METHOD_TEST_COMPLETE" in stdout, (
        f"R script did not complete. Output:\n{stdout[-1000:]}"
    )
    term_lines = re.findall(
        r"TERM=(\S+) REF=(\S+) MANUAL=(\S+) DIFF=(\S+)", stdout
    )
    assert len(term_lines) >= 2, (
        f"Expected at least 2 term comparisons, got {len(term_lines)}"
    )
    for term, ref_str, manual_str, diff_str in term_lines:
        diff = float(diff_str)
        assert diff < 1e-6, (
            f"Delta method mismatch for {term}: "
            f"ref={ref_str}, manual={manual_str}, diff={diff_str}"
        )


# =====
# Structural validation of LaTeX output
# =====
class TestTableStructure:
    """Verify LaTeX table structure is well-formed."""

    def test_full_table_has_all_panels(self, logit_full_text):
        """Full table contains all three panel headers."""
        assert "Panel A: All Participants" in logit_full_text
        assert "Panel B: Second Sellers" in logit_full_text
        assert "Panel C: First Sellers" in logit_full_text

    def test_compact_table_has_all_panels(self, logit_compact_text):
        """Compact table contains all three panel headers."""
        assert "Panel A: All Participants" in logit_compact_text
        assert "Panel B: Second Sellers" in logit_compact_text
        assert "Panel C: First Sellers" in logit_compact_text

    def test_full_table_has_longtable(self, logit_full_text):
        """Full table uses longtable environment."""
        assert "\\begin{longtable}" in logit_full_text
        assert "\\end{longtable}" in logit_full_text

    def test_full_table_has_pseudo_r2(self, logit_full_text):
        """Full table reports Pseudo R-squared (logit-specific)."""
        assert "Pseudo R$^2$" in logit_full_text

    def test_full_table_has_log_likelihood(self, logit_full_text):
        """Full table reports log-likelihood (logit-specific)."""
        assert "Log-likelihood" in logit_full_text

    def test_column_headers_say_logit(self, logit_full_text):
        """Column headers identify RE Logit and FE Logit."""
        assert "RE Logit" in logit_full_text
        assert "FE Logit" in logit_full_text

    def test_compact_has_appendix_reference(self, logit_compact_text):
        """Compact table references the full appendix table."""
        assert "unified_selling_logit_full" in logit_compact_text


# %%
if __name__ == "__main__":
    main()
