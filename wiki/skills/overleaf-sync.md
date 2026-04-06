---
title: "Overleaf Sync Workflow"
type: skill
tags: [overleaf, git, latex, paper, sync]
summary: "How to pull and push paper changes between the local repo and Overleaf using the configured git remote"
status: active
last_verified: "2026-04-06"
---

## Summary

The paper directory (`analysis/paper/`) syncs to Overleaf via a configured `overleaf` git remote. Changes push automatically on merge to main via GitHub Action, and can be pulled manually using `git fetch overleaf`.

## Key Points

- The repo has an `overleaf` remote with credentials stored in git's credential helper — no token needed
- **Push**: automatic via `.github/workflows/sync-overleaf.yml` on push to main
- **Pull**: manual using `git fetch overleaf` then rsync or cherry-pick
- After pulling, fix any `\includegraphics` or `\input` paths that Overleaf editors may have prefixed with `plots/` or `tables/` — locally these must be bare filenames

## Pulling from Overleaf

```bash
# Fetch latest from Overleaf
git fetch overleaf

# Compare a specific file
git show overleaf/master:main.tex > /tmp/overleaf_main.tex
diff analysis/paper/main.tex /tmp/overleaf_main.tex

# Bulk sync (rsync approach)
git clone https://git.overleaf.com/<project-id> /tmp/overleaf-pull
rsync -av --exclude='.git/' --exclude='tables/' --exclude='plots/' /tmp/overleaf-pull/ analysis/paper/
```

## Path Fixing After Pull

Overleaf editors sometimes add directory prefixes to includes. These break locally because `\graphicspath` and `\input@path` already include the prefix directories.

```latex
% Wrong (breaks locally):
\includegraphics{plots/welfare_theory}
\input{tables/h2_regression_cluster}

% Correct (bare filenames):
\includegraphics{welfare_theory}
\input{h2_regression_cluster}
```

## Related

- [Project Architecture](wiki/tools/architecture.md)
