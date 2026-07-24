"""Find LegalHK court-reasoning examples that motivate adaptive trajectories.

This is a read-only dataset analysis script. It scans the raw LegalHK parquet
for court-reasoning language that suggests different high-level reasoning
templates, then exports a concise Markdown report and CSV evidence table.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "raw" / "legalhk" / "train.parquet"
OUT_DIR = ROOT / "reports" / "legal_flux"
OUT_MD = OUT_DIR / "legalhk_court_reasoning_motivation_examples.md"
OUT_CSV = OUT_DIR / "legalhk_court_reasoning_motivation_examples.csv"


PATTERNS: dict[str, list[str]] = {
    "procedural_gateway": [
        r"\bsummary judgment\b",
        r"\bstrike out\b",
        r"\bleave to appeal\b",
        r"\bout of time\b",
        r"\bstay\b",
        r"\bjurisdiction\b",
        r"\barguable\b",
        r"\breal prospect\b",
        r"\bbona fide defence\b",
        r"\btriable issue\b",
        r"\babuse of process\b",
    ],
    "injunction_discretion": [
        r"\binjunction\b",
        r"\bserious question\b",
        r"\bbalance of convenience\b",
        r"\badequacy of damages\b",
        r"\bundertaking as to damages\b",
        r"\binterlocutory relief\b",
    ],
    "evidence_credibility_burden": [
        r"\bcredib",
        r"\breliable\b",
        r"\bI accept\b",
        r"\bI do not accept\b",
        r"\bwitness\b",
        r"\bburden\b",
        r"\bbalance of probabilities\b",
        r"\bproved\b",
        r"\bsatisfied\b",
    ],
    "statutory_or_rule_application": [
        r"\bsection\b",
        r"\bordinance\b",
        r"\brule\b",
        r"\bstatutory\b",
        r"\bpursuant to\b",
        r"\bconstruction\b",
        r"\binterpret",
        r"\bmeaning of\b",
    ],
    "precedent_analogy": [
        r"\bauthorit",
        r"\bprecedent\b",
        r"\bfollow",
        r"\bdistinguish",
        r"\bheld that\b",
        r"\bprinciple",
        r"\bcase law\b",
    ],
    "multi_issue_composition": [
        r"\bfirst\b",
        r"\bsecond\b",
        r"\bthird\b",
        r"\bissue\b",
        r"\bquestion\b",
        r"\btherefore\b",
        r"\baccordingly\b",
    ],
    "remedy_or_discretion": [
        r"\bdiscretion\b",
        r"\bdamages\b",
        r"\bspecific performance\b",
        r"\bdeclaration\b",
        r"\baccount\b",
        r"\bcosts\b",
        r"\brelief\b",
        r"\border\b",
    ],
}


EXAMPLE_CASE_IDS = [
    "legalhk-1864",
    "legalhk-9741",
    "legalhk-11593",
    "legalhk-6761",
    "legalhk-13093",
    "legalhk-6355",
]


def text_value(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_snippet(text: str, regexes: Iterable[str], width: int = 260) -> str:
    clean = compact(text)
    lower = clean.lower()
    for regex in regexes:
        match = re.search(regex, lower, flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - width // 2)
            end = min(len(clean), match.end() + width // 2)
            snippet = clean[start:end]
            if start:
                snippet = "..." + snippet
            if end < len(clean):
                snippet += "..."
            return snippet
    return clean[: width * 2] + ("..." if len(clean) > width * 2 else "")


def has_pattern(text: str, regexes: Iterable[str]) -> bool:
    lower = text.lower()
    return any(re.search(regex, lower, flags=re.IGNORECASE) for regex in regexes)


def issue_count(raw_issues: str) -> int:
    parts = [
        p.strip()
        for p in re.split(r"\n+|;(?=\s*(?:whether|if|what|the)\b)", raw_issues, flags=re.I)
        if p and p.strip()
    ]
    return max(1, len(parts)) if raw_issues.strip() else 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW_PARQUET)
    df = df.reset_index().rename(columns={"index": "row_index"})
    df["case_id"] = df["row_index"].map(lambda i: f"legalhk-{i}")
    df["court_reasoning_text"] = df["court_reasoning"].map(text_value)
    df["issues_text"] = df["issues"].map(text_value)
    df["issue_count_est"] = df["issues_text"].map(issue_count)
    df["reasoning_chars"] = df["court_reasoning_text"].map(len)
    df["related_law_count"] = df["related_laws"].map(lambda x: len([p for p in text_value(x).splitlines() if p.strip()]))
    df["relevant_case_count"] = df["relevant_cases"].map(lambda x: len([p for p in text_value(x).splitlines() if p.strip()]))

    for name, regexes in PATTERNS.items():
        df[name] = df["court_reasoning_text"].map(lambda text, r=regexes: has_pattern(text, r))

    pattern_counts = {name: int(df[name].sum()) for name in PATTERNS}
    combo_counter = Counter(
        "|".join(name for name in PATTERNS if row[name]) or "none"
        for _, row in df.iterrows()
    )

    rows = []
    for case_id in EXAMPLE_CASE_IDS:
        row = df.loc[df["case_id"] == case_id]
        if row.empty:
            continue
        r = row.iloc[0]
        matched = [name for name in PATTERNS if bool(r[name])]
        snippet = first_snippet(
            r["court_reasoning_text"],
            [regex for name in matched for regex in PATTERNS[name]] or [r"."],
            width=320,
        )
        rows.append(
            {
                "case_id": case_id,
                "lawsuit_type": text_value(r["lawsuit_type"]),
                "gold_label": text_value(r["support&reject"]),
                "claim": compact(text_value(r["plaintiff_claim"]))[:300],
                "issue_count_est": int(r["issue_count_est"]),
                "related_law_count": int(r["related_law_count"]),
                "relevant_case_count": int(r["relevant_case_count"]),
                "matched_patterns": "; ".join(matched),
                "court_reasoning_snippet": snippet,
                "issues": compact(text_value(r["issues"]))[:600],
            }
        )

    # Add high-signal examples for patterns not covered by the hand-picked set.
    covered = set()
    for row in rows:
        covered.update(row["matched_patterns"].split("; "))
    for pattern in PATTERNS:
        if pattern in covered:
            continue
        candidates = df[df[pattern]].copy()
        if candidates.empty:
            continue
        candidates["signal"] = (
            candidates["reasoning_chars"].clip(upper=4000)
            + candidates["issue_count_est"] * 200
            + candidates["related_law_count"] * 80
            + candidates["relevant_case_count"] * 80
        )
        r = candidates.sort_values("signal", ascending=False).iloc[0]
        matched = [name for name in PATTERNS if bool(r[name])]
        rows.append(
            {
                "case_id": r["case_id"],
                "lawsuit_type": text_value(r["lawsuit_type"]),
                "gold_label": text_value(r["support&reject"]),
                "claim": compact(text_value(r["plaintiff_claim"]))[:300],
                "issue_count_est": int(r["issue_count_est"]),
                "related_law_count": int(r["related_law_count"]),
                "relevant_case_count": int(r["relevant_case_count"]),
                "matched_patterns": "; ".join(matched),
                "court_reasoning_snippet": first_snippet(r["court_reasoning_text"], PATTERNS[pattern], width=320),
                "issues": compact(text_value(r["issues"]))[:600],
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("# LegalHK Court-Reasoning Examples for LegalFlux Motivation")
    lines.append("")
    lines.append("## Dataset Signals")
    lines.append("")
    lines.append(f"- Rows inspected: {len(df):,}")
    lines.append(f"- Non-empty court reasoning rows: {int(df['court_reasoning_text'].str.strip().ne('').sum()):,}")
    lines.append(f"- Rows with supplied related laws: {int((df['related_law_count'] > 0).sum()):,}")
    lines.append(f"- Rows with supplied relevant cases: {int((df['relevant_case_count'] > 0).sum()):,}")
    lines.append(f"- Rows with estimated multiple issues: {int((df['issue_count_est'] > 1).sum()):,}")
    lines.append("")
    lines.append("## Regex Pattern Counts")
    lines.append("")
    for name, count in sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{name}`: {count:,}")
    lines.append("")
    lines.append("## Most Frequent Pattern Combinations")
    lines.append("")
    for combo, count in combo_counter.most_common(12):
        lines.append(f"- `{combo}`: {count:,}")
    lines.append("")
    lines.append("## Candidate Introduction Examples")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['case_id']} — {row['lawsuit_type'] or '(blank lawsuit_type)'}")
        lines.append("")
        lines.append(f"- Gold label: `{row['gold_label']}`")
        lines.append(f"- Estimated issues: {row['issue_count_est']}; related laws: {row['related_law_count']}; relevant cases: {row['relevant_case_count']}")
        lines.append(f"- Matched high-level patterns: `{row['matched_patterns']}`")
        lines.append(f"- Claim: {row['claim']}")
        lines.append(f"- Issues: {row['issues']}")
        lines.append(f"- Court-reasoning excerpt: “{row['court_reasoning_snippet']}”")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print("Pattern counts:")
    for name, count in sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
