#!/usr/bin/env python3
"""Post daily arXiv Numerical Analysis papers to a GitHub issue.

This script fetches new submissions from the arXiv API in the Numerical
Analysis category (math.NA) and creates an issue in the repository where it is
run.  The issue contains a list of papers with short summaries extracted from
their abstracts.

The script expects the ``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY`` environment
variables to be set.  ``GITHUB_REPOSITORY`` is automatically provided inside
GitHub Actions runners and should have the form ``owner/repo``.

Usage:
    python post_arxiv_na_issue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Tuple

ARXIV_API = (
    "http://export.arxiv.org/api/query?search_query=cat:math.NA&"
    "sortBy=submittedDate&sortOrder=descending&max_results=50"
)
USER_AGENT = "arxiv-na-issue-bot/0.1 (+https://github.com)"


def fetch_daily_entries() -> List[Tuple[str, str, str]]:
    """Return a list of (title, short_summary, link) for new submissions."""
    req = urllib.request.Request(ARXIV_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    root = ET.fromstring(data)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=1)
    results: List[Tuple[str, str, str]] = []
    for entry in root.findall("atom:entry", ns):
        published = entry.find("atom:published", ns).text
        published_dt = dt.datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
        if published_dt < cutoff:
            continue
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        summary_short = shorten_summary(summary)
        link = entry.find("atom:id", ns).text.strip()
        results.append((title, summary_short, link))
    return results


def shorten_summary(text: str, max_sentences: int = 2) -> str:
    """Return the first ``max_sentences`` sentences from ``text``."""
    sentences = text.split(". ")
    clipped = ". ".join(sentences[:max_sentences]).strip()
    if not clipped.endswith('.'):
        clipped += '.'
    return clipped


def create_issue(title: str, body: str) -> None:
    """Create a GitHub issue using the REST API."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPOSITORY must be set")
    api = f"https://api.github.com/repos/{repo}/issues"
    payload = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(
        api,
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print instead of posting")
    args = parser.parse_args(argv)

    entries = fetch_daily_entries()
    if not entries:
        print("No new submissions found.")
        return 0

    today = dt.date.today().isoformat()
    issue_title = f"Daily arXiv NA papers: {today}"
    lines = [
        f"- [{title}]({link}): {summary}" for title, summary, link in entries
    ]
    issue_body = "\n".join(lines)

    if args.dry_run:
        print(issue_title)
        print(issue_body)
    else:
        create_issue(issue_title, issue_body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
