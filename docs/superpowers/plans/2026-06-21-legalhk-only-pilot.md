# LegalHK-Only Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing mixed-dataset experiment into an isolated LegalHK-only pilot, then complete and score a five-case smoke run without starting the 64-case main generation.

**Architecture:** A new configuration isolates all LegalHK-only artifacts while the preparation layer deterministically screens and splits LegalHK rows. Job construction consumes explicit smoke/evaluation split metadata, and scoring/reporting add binary and paired LegalHK metrics. Existing mixed-pilot artifacts remain readable and untouched.

**Tech Stack:** Python 3.11, Pydantic 2, pandas, NumPy, SciPy, scikit-learn, PyArrow, Ollama, pytest, YAML, JSONL.

---

The workspace is not a Git repository. Where the normal workflow would commit,
create a verification checkpoint by running the named focused and full tests and
recording the result in the task commentary.

## File map

- Create `configs/legalhk_only.yaml`: isolated experiment settings and paths.
- Create `src/legal_pilot/legalhk_selection.py`: pure civil/leakage screening,
  deterministic split selection, and review-record construction.
- Modify `src/legal_pilot/data_prep.py`: configuration-driven dataset
  preparation and LegalHK-only artifacts.
- Modify `src/legal_pilot/jobs.py`: split-aware smoke and evaluation jobs.
- Modify `src/legal_pilot/models.py`: LegalHK-only output schemas without
  `task_answer`.
- Modify `src/legal_pilot/runner.py`: final-decision-only normalization and
  split-aware generation.
- Modify `src/legal_pilot/scoring.py`: binary outcome and grounding metrics.
- Modify `src/legal_pilot/evaluation.py`: prediction recording and aggregate
  binary metrics.
- Modify `src/legal_pilot/reporting.py`: paired bootstrap and McNemar exports
  without OpenExempt assumptions.
- Modify prompt and JSON schema files: remove non-binary task-answer language.
- Modify `README.md`: document the LegalHK-only commands, counts, and limits.
- Create focused tests for each changed behavior before implementation.

### Task 1: Isolated LegalHK-only configuration

**Files:**
- Create: `configs/legalhk_only.yaml`
- Create: `tests/test_legalhk_config.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from legal_pilot.config import load_config, resolve_path


def test_legalhk_config_is_isolated_and_has_expected_counts():
    config = load_config("configs/legalhk_only.yaml")

    assert config["data"]["datasets"] == ["legalhk"]
    assert config["data"]["smoke_cases"] == 5
    assert config["data"]["evaluation_cases"] == 64
    assert config["data"]["oracle_cases"] == 24
    assert config["data"]["sampling_control_cases"] == 12
    assert config["data"]["sampling_control_repeats"] == 3
    assert str(resolve_path(config, "processed_dir")).endswith(
        r"data\processed\legalhk_only"
    )
    assert str(resolve_path(config, "runs_dir")).endswith(
        r"runs\legalhk_only"
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_legalhk_config.py -q
```

Expected: FAIL because `configs/legalhk_only.yaml` does not exist.

- [ ] **Step 3: Add the LegalHK-only YAML**

Use the existing model and audit settings, then set:

```yaml
project:
  seed: 20260619
  run_name: diagnostic

data:
  datasets: [legalhk]
  smoke_cases: 5
  evaluation_cases: 64
  oracle_cases: 24
  sampling_control_cases: 12
  sampling_control_repeats: 3
  max_input_characters: 48000
  decision_overlap_ngram: 6
  decision_overlap_threshold: 0.12

paths:
  raw_dir: data/raw
  processed_dir: data/processed/legalhk_only
  runs_dir: runs/legalhk_only
  reports_dir: reports/legalhk_only
  prompts_dir: prompts
  schemas_dir: schemas
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: PASS.

### Task 2: Deterministic civil and leakage screening

**Files:**
- Create: `src/legal_pilot/legalhk_selection.py`
- Create: `tests/test_legalhk_selection.py`

- [ ] **Step 1: Write failing tests for explicit leakage**

```python
from legal_pilot.legalhk_selection import (
    explicit_leakage_reasons,
    is_civil_legalhk_row,
)


def test_explicit_holding_and_credibility_language_are_rejected():
    text = (
        "The court held the defendant liable. "
        "The plaintiff was a truthful and reliable witness."
    )

    reasons = explicit_leakage_reasons(
        text,
        judgment_decision="The defendant is liable.",
        ngram_size=6,
        overlap_threshold=0.12,
    )

    assert "explicit_court_outcome" in reasons
    assert "credibility_finding" in reasons


def test_neutral_procedural_and_event_facts_pass():
    reasons = explicit_leakage_reasons(
        "The parties signed a lease in 2018. Rent was unpaid for three months.",
        judgment_decision="The claim is dismissed.",
        ngram_size=6,
        overlap_threshold=0.12,
    )

    assert reasons == []


def test_hksar_and_sentencing_rows_are_not_civil():
    assert not is_civil_legalhk_row(
        plaintiff="HKSAR",
        lawsuit_type="criminal case",
        claim="trafficking in a dangerous drug",
    )
    assert is_civil_legalhk_row(
        plaintiff="Alice",
        lawsuit_type="negligence claim",
        claim="damages for vehicle repair",
    )
```

- [ ] **Step 2: Run tests and verify RED**

Expected: import failure because `legalhk_selection.py` does not exist.

- [ ] **Step 3: Implement the minimal screening module**

Define compiled patterns for:

```python
OUTCOME_PATTERNS = {
    "explicit_court_outcome": (
        r"\bthe (?:court|tribunal|judge) "
        r"(?:held|found|concluded|determined|ordered|awarded|dismissed|"
        r"allowed|granted|rejected|ruled|decided)\b"
    ),
    "claim_disposition": (
        r"\b(?:claim|claims|appeal|application|counterclaim) "
        r"(?:was|were|is|are|be) "
        r"(?:dismissed|allowed|granted|rejected|struck out)\b"
    ),
    "liability_conclusion": (
        r"\b(?:plaintiff|defendant|respondent|appellant) "
        r"(?:was|is|were|are) (?:not )?liable\b"
    ),
    "award_or_order": (
        r"\b(?:shall pay|awarded damages|damages (?:of|in the sum of)|"
        r"judgment (?:for|against)|final order)\b"
    ),
    "credibility_finding": (
        r"\b(?:truthful|reliable|evasive|unreliable|credible) witness\b|"
        r"\b(?:accepted|rejected) (?:the )?(?:evidence|testimony)\b"
    ),
}
```

Implement normalized token n-grams and add
`"judgment_text_overlap"` when the fraction of decision n-grams present in the
fact text reaches the configured threshold. Implement civil exclusion using
HKSAR and criminal-domain terms.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: PASS.

- [ ] **Step 5: Add failing tests for balanced disjoint splits**

Build a synthetic pandas frame with at least 40 rows per class and assert:

```python
smoke, evaluation = select_legalhk_splits(
    frame,
    smoke_count=5,
    evaluation_count=64,
    seed=20260619,
)
assert len(smoke) == 5
assert len(evaluation) == 64
assert set(smoke.index).isdisjoint(evaluation.index)
assert evaluation["support&reject"].value_counts().to_dict() == {
    "support": 32,
    "reject": 32,
}
```

- [ ] **Step 6: Verify RED, implement deterministic round-robin stratification, and verify GREEN**

Create strata from issue bucket, length bucket, defense presence, and lawsuit
type. Shuffle each stratum with a local seeded random generator and select
round-robin within each outcome class. Select evaluation first, then smoke from
the remaining rows so the evaluation balance is exact.

### Task 3: LegalHK-only preparation artifacts

**Files:**
- Modify: `src/legal_pilot/data_prep.py`
- Modify: `src/legal_pilot/models.py`
- Create: `tests/test_legalhk_preparation.py`

- [ ] **Step 1: Write a failing preparation test using a temporary parquet**

Monkeypatch `download_file` to leave a synthetic parquet in place. Assert that
`prepare_datasets(config)` writes:

```python
assert manifest["datasets"] == ["legalhk"]
assert manifest["smoke_cases"] == 5
assert manifest["evaluation_cases"] == 64
assert manifest["total_cases"] == 69
assert manifest["evaluation_outcomes"] == {"reject": 32, "support": 32}
assert manifest["smoke_evaluation_overlap"] == 0
assert manifest["leakage_screen"]["excluded_rows"] > 0
```

Load `cases.jsonl` and assert every case has
`metadata["selection_split"]` equal to `smoke` or `evaluation`. Load
`selection_review.jsonl` and assert it contains no keys named `gold_answer`,
`court_reasoning`, `judgment_decision`, or `support&reject`.

- [ ] **Step 2: Run and verify RED**

Expected: current preparation downloads OpenExempt and lacks split artifacts.

- [ ] **Step 3: Make preparation configuration-driven**

In `prepare_datasets`, call OpenExempt only when `"openexempt"` appears in
`config["data"]["datasets"]`. For LegalHK-only, call the revised
`prepare_legalhk` with smoke/evaluation counts and leakage settings.

Return selected cases plus a selection manifest and review rows. Write:

```text
data/processed/legalhk_only/cases.jsonl
data/processed/legalhk_only/prepare_manifest.json
data/processed/legalhk_only/selection_review.jsonl
```

Normalize LegalHK cases with:

```python
authorities=None
variant_id="original"
metadata={
    "selection_split": split,
    "lawsuit_type": row["lawsuit_type"],
    "issue_count": len(issues),
    "fact_characters": len(row["more_facts"]),
    "has_defense": bool(row["has_defense"]),
    "leakage_screen": "auto_pass",
    "license_warning": "processed release license unknown",
}
```

Do not store court reasoning or judgment decision in normalized case metadata.
Construct `reference_state` before discarding `related_laws`, using sanitized
issues and the related-law text as the rule/test.

- [ ] **Step 4: Restrict model outputs to binary LegalHK decisions**

Remove `task_answer` from `DirectAnalysis` and `FinalAnalysis`. Keep
`FinalDecisionValue` schema-compatible but ensure scoring treats only support
and reject as valid predictions. Retain both dataset literals temporarily so
old stored records can still be parsed by utility code.

- [ ] **Step 5: Run focused preparation tests and verify GREEN**

- [ ] **Step 6: Run all tests as a checkpoint**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass after updating obsolete OpenExempt-specific tests in
later tasks; any immediate failures must be understood and recorded.

### Task 4: Split-aware job construction and exact run counts

**Files:**
- Modify: `src/legal_pilot/jobs.py`
- Create: `tests/test_legalhk_jobs.py`

- [ ] **Step 1: Write failing count and isolation tests**

Construct five smoke and 64 evaluation `NormalizedCase` objects and assert:

```python
smoke_jobs = build_jobs(cases, config, smoke=True)
main_jobs = build_jobs(cases, config, smoke=False)

assert len(smoke_jobs) == 20
assert {job["case"].metadata["selection_split"] for job in smoke_jobs} == {
    "smoke"
}
assert len(main_jobs) == 316
assert {job["case"].metadata["selection_split"] for job in main_jobs} == {
    "evaluation"
}
assert sum(job["condition"] == "oracle" for job in main_jobs) == 24
assert sum(job["condition"] == "sampling_control" for job in main_jobs) == 36
```

- [ ] **Step 2: Run and verify RED**

Expected: the current code reads obsolete OpenExempt count keys.

- [ ] **Step 3: Implement split-aware jobs**

Select cases by `metadata["selection_split"]`. Add four principal jobs per
selected case. For non-smoke runs, append Oracle jobs from the first 24
evaluation cases and three sampling-control jobs for each of the first 12
evaluation cases. Preserve deterministic final shuffling.

- [ ] **Step 4: Run and verify GREEN**

### Task 5: LegalHK prompts, schemas, and runner

**Files:**
- Modify: `prompts/direct.txt`
- Modify: `prompts/structured.txt`
- Modify: `prompts/state_analysis.txt`
- Modify: `schemas/direct_analysis.json`
- Modify: `schemas/final_analysis.json`
- Modify: `src/legal_pilot/runner.py`
- Modify: `tests/test_output_normalization.py`

- [ ] **Step 1: Add failing schema and normalization tests**

Assert that both analysis schemas:

```python
assert "task_answer" not in schema["properties"]
assert set(schema["properties"]["final_decision"]["enum"]) == {
    "support", "reject", "mixed", "unresolved"
}
```

Assert `_normalize_direct_payload` and
`_normalize_final_analysis_payload` never synthesize `task_answer`.

- [ ] **Step 2: Run and verify RED**

- [ ] **Step 3: Remove non-binary instructions and schema fields**

Each prompt must state that the legal prediction is binary and that the model
should select `support` or `reject`; `mixed` and `unresolved` are allowed only
when the supplied facts genuinely prevent a binary resolution and will score as
incorrect. Remove every `task_answer` instruction.

Update runner output normalization, `FinalAnalysis` construction, and prediction
recording to use `final_decision` only.

- [ ] **Step 4: Run focused tests and verify GREEN**

### Task 6: Binary scoring and paired statistical comparisons

**Files:**
- Modify: `src/legal_pilot/scoring.py`
- Modify: `src/legal_pilot/evaluation.py`
- Modify: `src/legal_pilot/reporting.py`
- Modify: `tests/test_scoring.py`
- Create: `tests/test_legalhk_statistics.py`

- [ ] **Step 1: Write failing binary scoring tests**

```python
scores = score_record(case, analysis)
assert scores["answer_correct"] == 1.0
assert scores["binary_prediction_valid"] == 1.0
assert scores["conclusion_with_fact_rate"] == 1.0
```

Add a second analysis with `final_decision="unresolved"` and no fact IDs:

```python
assert scores["answer_correct"] == 0.0
assert scores["binary_prediction_valid"] == 0.0
assert scores["conclusion_with_fact_rate"] == 0.0
```

- [ ] **Step 2: Run and verify RED**

- [ ] **Step 3: Implement final-decision-only scoring**

Set `predicted = analysis.final_decision`. Add:

```python
binary_prediction_valid = float(predicted in {"support", "reject"})
conclusion_with_fact_rate = (
    mean(bool(c.supporting_fact_ids or c.opposing_fact_ids)
         for c in analysis.issue_conclusions)
    if analysis.issue_conclusions else 0.0
)
```

Retain valid-ID measures as secondary hygiene metrics.

- [ ] **Step 4: Add failing paired-statistics tests**

Create paired Structured and Validated rows with known outcomes and assert:

```python
comparison = paired_condition_comparisons(
    rows, baseline="structured", seed=20260619, samples=500
)
row = comparison.query("condition == 'validated'").iloc[0]
assert row["paired_n"] == 4
assert row["accuracy_difference"] == 0.25
assert row["baseline_only_correct"] == 0
assert row["condition_only_correct"] == 1
assert 0.0 <= row["mcnemar_exact_p"] <= 1.0
```

- [ ] **Step 5: Verify RED and implement paired statistics**

Join rows by dataset/case/variant. Treat non-OK records as incorrect. Bootstrap
the mean paired correctness difference. Compute exact McNemar p-values with:

```python
from scipy.stats import binomtest

p_value = (
    binomtest(min(b, c), n=b + c, p=0.5).pvalue
    if b + c else 1.0
)
```

Export `paired_condition_comparisons.csv`.

- [ ] **Step 6: Add macro-F1 and class recall**

Use `sklearn.metrics.f1_score` with labels `["support", "reject"]`,
`average="macro"`, and `zero_division=0`. Invalid/error predictions remain
invalid strings so they reduce recall and count as incorrect rather than being
silently dropped. Add support and reject recall to condition summaries.

- [ ] **Step 7: Run focused and full tests**

Expected: all tests pass.

### Task 7: Remove mixed-dataset report assumptions

**Files:**
- Modify: `src/legal_pilot/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing LegalHK-only report tests**

Create summary/audit fixtures containing only LegalHK and assert `_recommend`
returns a recommendation without indexing OpenExempt rows. Assert stratum
summary uses lawsuit type and the report includes the LegalHK construction
limitation.

- [ ] **Step 2: Run and verify RED**

Expected: current recommendation directly indexes OpenExempt metrics.

- [ ] **Step 3: Make report generation dataset-agnostic**

Remove hard-coded OpenExempt calculations and prose. Report:

- LegalHK principal accuracy, macro-F1, and failure rates;
- Typed/Validated/Oracle gaps versus Structured;
- paired confidence intervals and McNemar tests;
- grounding and issue-coverage audit differences;
- token/latency ratios;
- LegalHK leakage and licensing limitations.

Set the recommendation from LegalHK evidence only. Require both outcome and
substantive audit improvement before recommending scale-up.

- [ ] **Step 4: Run focused and full tests**

Expected: all tests pass.

### Task 8: Documentation and dry-run verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update commands**

Document:

```powershell
.\.venv\Scripts\python.exe -m legal_pilot --config configs/legalhk_only.yaml prepare
.\.venv\Scripts\python.exe -m legal_pilot --config configs/legalhk_only.yaml smoke --dry-run
.\.venv\Scripts\python.exe -m legal_pilot --config configs/legalhk_only.yaml smoke
.\.venv\Scripts\python.exe -m legal_pilot --config configs/legalhk_only.yaml score --smoke
.\.venv\Scripts\python.exe -m legal_pilot --config configs/legalhk_only.yaml freeze
.\.venv\Scripts\python.exe -m legal_pilot --config configs/legalhk_only.yaml generate --dry-run
```

State explicitly that the main `generate` command is not run during this task.

- [ ] **Step 2: Run the complete automated test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass with no warnings attributable to the changes.

- [ ] **Step 3: Run preparation and inspect invariants**

Run LegalHK-only `prepare`. Verify:

- 5 smoke and 64 evaluation cases;
- 69 unique case IDs;
- no split overlap;
- 32 support and 32 reject evaluation cases;
- all selected rows have zero leakage reasons;
- review file contains no prohibited label or judgment fields.

- [ ] **Step 4: Verify dry-run counts**

Expected:

```text
smoke --dry-run: 20 jobs
generate --dry-run: 316 jobs
```

### Task 9: Actual five-case smoke run

**Files:**
- Generated: `runs/legalhk_only/smoke/generations.jsonl`
- Generated: `runs/legalhk_only/smoke/scored.jsonl`
- Generated: `runs/legalhk_only/smoke/aggregate.csv`

- [ ] **Step 1: Check Ollama and model digest**

Run `env-check` with the LegalHK-only config. Confirm NVIDIA acceleration is
reported and the expected `qwen3.5:9b` digest is available.

- [ ] **Step 2: Run the 20-job smoke test**

Run `smoke` with the LegalHK-only config. Do not start `generate`.

- [ ] **Step 3: Verify smoke completeness**

Load the latest record per run hash and assert:

- 20 unique current hashes;
- five cases under each principal condition;
- 20 `status == "ok"` records;
- all parsed outputs validate;
- malformed attempts, if any, remain in history;
- Typed records use two calls;
- Validated records use two or three calls.

- [ ] **Step 4: Verify resume behavior**

Run `smoke` again and expect `completed: 0`, `skipped: 20`.

- [ ] **Step 5: Score smoke**

Run `score --smoke`. Verify 20 scored records and export the aggregate table.

- [ ] **Step 6: Inspect the five condition-blind case inputs and smoke outputs**

Check for residual explicit leakage, schema truncation, unsupported issue
invention, and accidental exposure of authorities/reference issues. Record any
problem before freezing.

- [ ] **Step 7: Stop before Phase 2 freeze or main generation**

Return the smoke counts, failures, latency, outcome results, and any prompt or
schema concerns to the user. Do not run the 316-job main experiment.
