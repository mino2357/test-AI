#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, textwrap, html, re
import feedparser
import requests
from datetime import datetime, timezone
from dateutil import parser as dtparser
from typing import List

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEARCH_QUERY = '(cat:math.NA OR cat:cs.NA)'
MAX_RESULTS = 200
LABELS = ["arxiv-na"]

# ---------- small utils ----------

def iso_ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def today_utc_ymd() -> str:
    return iso_ymd(datetime.now(timezone.utc))

def github_request(method: str, url: str, token: str, json=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.request(method, url, headers=headers, json=json, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text}")
    if resp.status_code == 204:
        return {}
    return resp.json()

# ---------- arXiv fetch & parse ----------

def fetch_entries():
    """ submittedDate 降順→ published(UTC) が“今日”のみ採用。重複ID排除。 """
    params = {
        "search_query": SEARCH_QUERY,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    time.sleep(1.0)  # polite
    q = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    feed = feedparser.parse(f"{ARXIV_API_URL}?{q}")
    entries = getattr(feed, "entries", [])
    today = today_utc_ymd()

    seen = set()
    todays = []
    for e in entries:
        pub = dtparser.parse(getattr(e, "published", ""))
        if iso_ymd(pub.astimezone(timezone.utc)) != today:
            continue
        eid = getattr(e, "id", "").strip()
        if eid and eid in seen:
            continue
        seen.add(eid)
        todays.append(e)
    return todays

def detect_categories(e) -> list:
    cats = []
    for t in getattr(e, "tags", []) or []:
        term = getattr(t, "term", "").strip()
        if term:
            cats.append(term)
    cats = [c for c in cats if c in ("math.NA", "cs.NA")]
    return cats or ["(unknown)"]

def format_authors(e) -> str:
    try:
        authors = [a.name for a in e.authors]
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + " et al."
    except Exception:
        return "Unknown"

def abs_url(e) -> str:
    return getattr(e, "id", "").strip()

def pdf_url_from_abs(abs_url: str) -> str:
    return re.sub(r"https?://arxiv\.org/abs/([^/\s]+)", r"https://arxiv.org/pdf/\1.pdf", abs_url)

def find_code_links(text: str) -> list:
    urls = re.findall(r"https?://[^\s)]+", text)
    code_like = [u for u in urls if any(s in u.lower() for s in ["github.com", "gitlab.com", "bitbucket.org", "codeocean", "zenodo.org/record"])]
    seen, out = set(), []
    for u in code_like:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:3]

# ---------- summarization & math extraction ----------

STOPWORDS = set("""
the of and to in for a an is are with on by from this that which be as we our their using used
method methods approach approaches result results show shows paper propose proposed new study
analysis numerical also can may based provide its it they them into model models problem problems
""".split())

def first_sentences(text: str, n=2) -> str:
    if not text: return ""
    t = text.replace("\n", " ").strip()
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+", t) if p.strip()]
    if len(parts) >= n:
        s = " ".join(parts[:n])
        if not s.endswith("."): s += "."
        return s
    return (t[:300] + ("…" if len(t) > 300 else ""))

def keyword_pick(text: str, k=6) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    freq = {}
    for w in words:
        if w in STOPWORDS or len(w) <= 2: continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]
    return ", ".join([w for w, _ in top])

MATH_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),          # $$ ... $$
    re.compile(r"(?<!\$)\$(.+?)(?<!\$)\$", re.DOTALL),# $ ... $（$$は避ける）
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),          # \( ... \)
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),          # \[ ... \]
]

def extract_first_math(tex: str) -> str:
    """抄録中の最初の数式を抽出（display 優先）。返り値は $$...$$ で包む。"""
    if not tex: return ""
    for pat in (MATH_PATTERNS[0], MATH_PATTERNS[3]):  # display 優先
        m = pat.search(tex)
        if m:
            body = m.group(1).strip()
            return f"$$\n{body}\n$$"
    for pat in (MATH_PATTERNS[1], MATH_PATTERNS[2]):  # inline
        m = pat.search(tex)
        if m:
            body = m.group(1).strip()
            return f"$$\n{body}\n$$"
    return ""

def rule_based_summary(abstract: str):
    """(tldr, key_math) を返す。"""
    tldr = first_sentences(abstract, n=2)
    keys = keyword_pick(abstract, k=6)
    if keys:
        tldr = (tldr + "\n" + f"Keywords: {keys}").strip()
    key_math = extract_first_math(abstract)
    return tldr, key_math

# ---------- Issue layout ----------

def badges(cats: List[str]) -> str:
    tags = []
    for c in cats:
        if c == "math.NA": tags.append("`math.NA`")
        elif c == "cs.NA": tags.append("`cs.NA`")
        else: tags.append(f"`{c}`")
    return " ".join(tags)

def build_full_list_section(entries):
    """その日の全タイトル＋リンク（Abs/PDF）を列挙。重複なく“全て”載せる。"""
    lines = []
    lines.append("## Full list (all titles & links)")
    lines.append("")
    for e in entries:
        title = (getattr(e, "title", "") or "").replace("\n", " ").strip()
        a = abs_url(e)
        p = pdf_url_from_abs(a)
        cats = badges(detect_categories(e))
        lines.append(f"- {cats} [{title}]({a}) · [PDF]({p})")
    lines.append("")  # 末尾空行
    return "\n".join(lines)

def build_issue_body(entries):
    today = today_utc_ymd()
    if not entries:
        return f"**Date (UTC)**: {today}\n\n本日の math.NA / cs.NA 新着は見つかりませんでした。"

    # math.NA を先に
    def sort_key(e):
        cats = detect_categories(e)
        return (0 if "math.NA" in cats else 1, getattr(e, "title", ""))

    entries = sorted(entries, key=sort_key)

    out = []
    out.append(f"# arXiv NA digest — {today} (UTC)")
    out.append("")
    out.append(f"- 件数: **{len(entries)}**")
    out.append("- チェック: ☐ skim ☐ read ☐ cite")
    out.append("")

    # 追加: “全てのタイトル＋リンク” セクション（常に先頭に配置）
    out.append(build_full_list_section(entries))

    # 各エントリの詳細
    for i, e in enumerate(entries, 1):
        title = (getattr(e, "title", "") or "").replace("\n", " ").strip()
        authors = format_authors(e)
        abslink = abs_url(e)
        pdflink = pdf_url_from_abs(abslink)
        abstract_raw = html.unescape(getattr(e, "summary", "").strip())
        cats = detect_categories(e)
        tldr, key_math = rule_based_summary(abstract_raw)
        code_links = find_code_links(abstract_raw)

        out.append(f"### {i}. {title}  {badges(cats)}")
        out.append("")
        out.append(f"- **Authors**: {authors}")
        links_line = [f"[Abs]({abslink})", f"[PDF]({pdflink})"]
        for j, url in enumerate(code_links, 1):
            links_line.append(f"[Code{j}]({url})")
        out.append("- " + " · ".join(links_line))
        out.append("")

        details = []
        details.append("<details><summary>Abstract & Summary（クリックで展開）</summary>\n")
        if tldr:
            details.append(f"**TL;DR**\n\n{tldr}\n")
        if abstract_raw:
            details.append("**Abstract (from arXiv)**\n")
            details.append(abstract_raw.strip() + "\n")
        if key_math:
            details.append("**Key formula**\n")
            details.append(key_math + "\n")
        details.append("</details>\n")
        out.append("\n".join(details))

    return "\n".join(out).strip()

# ---------- Issue upsert ----------

def find_issue_by_title(repo: str, token: str, title: str):
    url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page=50"
    data = github_request("GET", url, token)
    for it in data:
        if it.get("title") == title and "pull_request" not in it:
            return it
    return None

def create_or_update_issue(repo: str, token: str, title: str, body: str, labels=None):
    labels = labels or []
    existing = find_issue_by_title(repo, token, title)
    if existing:
        issue_url = existing["url"]
        current_labels = [l["name"] for l in existing.get("labels", [])]
        label_set = sorted(set(current_labels) | set(labels))
        github_request("PATCH", issue_url, token, json={"body": body, "labels": label_set})
        return existing["html_url"]
    else:
        url = f"https://api.github.com/repos/{repo}/issues"
        data = github_request("POST", url, token, json={"title": title, "body": body, "labels": labels})
        return data["html_url"]

# ---------- main ----------

def main():
    repo = os.environ.get("GITHUB_REPOSITORY")  # "mino2357/test-AI"
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY / GITHUB_TOKEN が未設定です。GitHub Actions で実行してください。", file=sys.stderr)
        sys.exit(2)

    entries = fetch_entries()
    body = build_issue_body(entries)
    title = f"arXiv NA - {today_utc_ymd()}"
    url = create_or_update_issue(repo, token, title, body, labels=LABELS)
    print(f"Done. Issue: {url}")

if __name__ == "__main__":
    main()
