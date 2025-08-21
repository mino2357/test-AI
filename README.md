# test-AI

AIサービスのテスト

This repository includes a GitHub Actions workflow that creates a daily issue
listing the latest [arXiv Numerical Analysis](https://arxiv.org/list/math.NA/new)
submissions with short summaries. The action runs once per day and publishes
the results to this repository's issues using the built-in `GITHUB_TOKEN`.

### Features

- Fetches both `math.NA` and `cs.NA` submissions from arXiv.
- Falls back to the previous day's feed if today's query returns no entries.
- Lists all titles with Abs/PDF links, followed by per-paper details.
- Shows a short TL;DR with keywords extracted from the abstract.
- Picks the most informative formula in the abstract (longest expression).
- Ensures formulas render correctly inside the expandable sections.

The `scripts/post_arxiv_na_issue.py` script powers the workflow and can be
run locally (requires `GITHUB_REPOSITORY` and `GITHUB_TOKEN` environment
variables).
