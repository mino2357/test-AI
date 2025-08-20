# test-AI

AIサービスのテスト

This repository includes a GitHub Actions workflow that creates a daily issue
listing the latest [arXiv Numerical Analysis](https://arxiv.org/list/math.NA/new)
submissions with short summaries. The action runs once per day and publishes
the results to this repository's issues using the built-in `GITHUB_TOKEN`.
