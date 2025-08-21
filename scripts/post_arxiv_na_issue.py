#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, textwrap, html, re, json
import urllib.request, urllib.parse, urllib.error
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from typing import List
import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEARCH_QUERY = '(cat:math.NA OR cat:cs.NA)'
MAX_RESULTS = 200
LABELS = ["arxiv-na"]

# ---------- small utils ----------

def iso_ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def today_utc_ymd() -> str:
    return iso_ymd(datetime.now(timezone.utc))

def github_request(method: str, url: str, token: str, json_data=None):
    data_bytes = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    if json_data is not None:
        data_bytes = json.dumps(json_data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.getcode()
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode()
    if status >= 300:
        raise RuntimeError(f"GitHub API error {status}: {body}")
    if status == 204:
        return {}
    return json.loads(body)

# ---------- arXiv fetch & parse ----------

def _parse_arxiv_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def fetch_entries(target_date: str = None):
    """submittedDate 降順→ published(UTC) が target_date のみ採用。重複ID排除。"""
    target_date = target_date or today_utc_ymd()
    params = {
        "search_query": SEARCH_QUERY,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    time.sleep(1.0)  # polite
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{ARXIV_API_URL}?{q}",
        headers={"User-Agent": "arxiv-na-script (mailto:example@example.com)"},
    )
    with urllib.request.urlopen(req) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    seen = set()
    todays = []
    for node in root.findall("atom:entry", ns):
        pub_text = node.findtext("atom:published", default="", namespaces=ns)
        pub = _parse_arxiv_date(pub_text)
        if iso_ymd(pub) != target_date:
            continue
        eid = node.findtext("atom:id", default="", namespaces=ns).strip()
        if eid and eid in seen:
            continue
        seen.add(eid)
        title = node.findtext("atom:title", default="", namespaces=ns)
        summary = node.findtext("atom:summary", default="", namespaces=ns)
        authors = []
        for a in node.findall("atom:author", ns):
            name = a.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(SimpleNamespace(name=name))
        tags = []
        for c in node.findall("atom:category", ns):
            term = c.attrib.get("term", "").strip()
            if term:
                tags.append(SimpleNamespace(term=term))
        todays.append(
            SimpleNamespace(
                id=eid,
                title=title,
                summary=summary,
                published=pub_text,
                authors=authors,
                tags=tags,
            )
        )
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

def extract_main_math(tex: str) -> str:
    """抄録中から情報量の多い数式（文字数最大）を抽出。返り値は $$...$$ で包む。"""
    if not tex:
        return ""

    candidates = []
    # display 数式を優先的に収集
    for pat in (MATH_PATTERNS[0], MATH_PATTERNS[3]):
        for m in pat.finditer(tex):
            candidates.append(m.group(1).strip())

    # display が見つからなければ inline
    if not candidates:
        for pat in (MATH_PATTERNS[1], MATH_PATTERNS[2]):
            for m in pat.finditer(tex):
                candidates.append(m.group(1).strip())

    if not candidates:
        return ""

    def info_len(s: str) -> int:
        return len(re.sub(r"\s+", "", s))

    best = max(candidates, key=info_len)
    return f"$$\n{best}\n$$"

def rule_based_summary(abstract: str):
    """(tldr, key_math) を返す。"""
    tldr = first_sentences(abstract, n=2)
    keys = keyword_pick(abstract, k=6)
    if keys:
        tldr = (tldr + "\n" + f"Keywords: {keys}").strip()
    key_math = extract_main_math(abstract)
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

def build_issue_body(entries, date_str: str):
    if not entries:
        return f"**Date (UTC)**: {date_str}\n\n本日の math.NA / cs.NA 新着は見つかりませんでした。"

    # math.NA を先に
    def sort_key(e):
        cats = detect_categories(e)
        return (0 if "math.NA" in cats else 1, getattr(e, "title", ""))

    entries = sorted(entries, key=sort_key)

    out = []
    out.append(f"# arXiv NA digest — {date_str} (UTC)")
    out.append("")
    out.append(f"- 件数: **{len(entries)}**")
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
        details.append("<details><summary>Abstract & Summary（クリックで展開）</summary>\n\n")
        if tldr:
            details.append(f"**TL;DR**\n\n{tldr}\n")
        if abstract_raw:
            details.append("**Abstract (from arXiv)**\n")
            details.append(abstract_raw.strip() + "\n")
        if key_math:
            details.append("**Key formula**\n\n")
            details.append(key_math + "\n")
        details.append("</details>\n")
        out.append("\n".join(details))

    return "\n".join(out).strip()

# ---------- Issue upsert ----------

def find_issue_by_title(repo: str, token: str, title: str):
    """Return the issue dict matching *title* if it exists.

    The repository might have many issues/PRs, so iterate through pages
    instead of only checking the first page. Stops after a page returns
    fewer than 1 item (i.e., end of results).
    """
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100&page={page}"
        data = github_request("GET", url, token)
        if not data:
            return None
        for it in data:
            if it.get("title") == title and "pull_request" not in it:
                return it
        page += 1

def ensure_labels_exist(repo: str, token: str, labels):
    for name in labels:
        url = f"https://api.github.com/repos/{repo}/labels/{urllib.parse.quote(name)}"
        try:
            github_request("GET", url, token)
        except RuntimeError as e:
            if "404" in str(e):
                github_request(
                    "POST",
                    f"https://api.github.com/repos/{repo}/labels",
                    token,
                    json_data={"name": name, "color": "c5def5"},
                )
            else:
                raise

def create_or_update_issue(repo: str, token: str, title: str, body: str, labels=None):
    labels = labels or []
    ensure_labels_exist(repo, token, labels)
    existing = find_issue_by_title(repo, token, title)
    if existing:
        issue_url = existing["url"]
        current_labels = [l["name"] for l in existing.get("labels", [])]
        label_set = sorted(set(current_labels) | set(labels))
        github_request("PATCH", issue_url, token, json_data={"body": body, "labels": label_set})
        return existing["html_url"]
    else:
        url = f"https://api.github.com/repos/{repo}/issues"
        data = github_request("POST", url, token, json_data={"title": title, "body": body, "labels": labels})
        return data["html_url"]

# ---------- main ----------

def main():
    repo = os.environ.get("GITHUB_REPOSITORY")  # "mino2357/test-AI"
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("GITHUB_REPOSITORY / GITHUB_TOKEN が未設定です。GitHub Actions で実行してください。", file=sys.stderr)
        sys.exit(2)

    today = today_utc_ymd()
    target_date = today
    entries = fetch_entries(target_date)
    if not entries:
        target_date = iso_ymd(datetime.now(timezone.utc) - timedelta(days=1))
        entries = fetch_entries(target_date)

    body = build_issue_body(entries, target_date)
    title = f"arXiv NA - {target_date}"
    url = create_or_update_issue(repo, token, title, body, labels=LABELS)
    print(f"Done. Issue: {url}")

if __name__ == "__main__":
    main()
