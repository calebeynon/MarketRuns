"""
Purpose: Validate Cox reactive table notes against the restricted-risk-set methodology (issue #118).
Author: Codex
Date: 2026-04-22
"""

from pathlib import Path

import pytest

# FILE PATHS
REPO_ROOT = Path(__file__).resolve().parents[2]
NORMAL_VS_REACTIVE_TEX = (
    REPO_ROOT / "analysis" / "output" / "tables" / "cox_survival_normal_vs_reactive.tex"
)
REACTIVE_500MS_TEX = (
    REPO_ROOT / "analysis" / "output" / "tables" / "cox_survival_reactive_500ms.tex"
)


# =====
# Main
# =====
def main():
    """Run all tests with verbose output."""
    pytest.main([__file__, "-v"])


# =====
# Helpers
# =====
def _read_text(path):
    """Fail fast when an expected table artifact is missing."""
    if not path.exists():
        pytest.fail(f"Missing table output: {path}")
    return path.read_text()


# =====
# Tests
# =====
def test_normal_vs_reactive_note_uses_restricted_risk_set_language():
    """Side-by-side table note must describe the restricted risk set explicitly."""
    tex = _read_text(NORMAL_VS_REACTIVE_TEX)
    assert "restricted to rows with group\\_sold\\_prev\\_period $=$ 1" in tex
    assert "event is any sale within that restricted risk set" in tex


def test_reactive_500ms_note_drops_pre_restriction_event_language():
    """Reactive-only table note must not describe non-reactive sales in the risk set."""
    tex = _read_text(REACTIVE_500MS_TEX)
    assert "restricted risk set" in tex
    assert "Non-reactive sales included in risk set as non-events." not in tex


# %%
if __name__ == "__main__":
    main()
