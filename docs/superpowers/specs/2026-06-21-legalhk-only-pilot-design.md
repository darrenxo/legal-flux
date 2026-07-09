# LegalHK-Only Case-State Pilot Design

## Objective

Revise the existing diagnostic pilot so that its next experiment uses only
LegalHK civil cases. The experiment will test whether constructing a typed or
validated legal case state improves binary support/reject judgment prediction
over direct and ordinary structured analysis.

The previous OpenExempt and mixed-dataset artifacts remain untouched. The new
configuration writes to separate processed-data, run, and report directories.

## Experimental stages

### Smoke stage

- Select five LegalHK cases used only for smoke testing.
- Run Direct, Structured, Typed, and Validated conditions.
- Produce 20 unique generation hashes.
- Verify structured parsing, scoring, checkpoint/resume behavior, validation,
  and malformed-output retention.
- Do not include smoke cases in the evaluation set.

### Diagnostic main stage

- Select 64 separate LegalHK evaluation cases: 32 support and 32 reject.
- Run four principal conditions on every case: 256 hashes.
- Run Oracle on 24 evaluation cases: 24 hashes.
- Run three temperature-0.7 Structured samples on 12 cases: 36 hashes.
- Total: 316 unique generation hashes and approximately 444–508 model calls,
  depending on Validated repair calls.
- Do not start this stage as part of the pipeline-update task.

### Later confirmation

If the 64-case diagnostic gives a meaningful signal, select a new confirmation
set of approximately 192–256 LegalHK cases. The confirmation set must not reuse
smoke or diagnostic cases.

## Dataset selection

Use the locally cached LegalHK parquet file. Retain only:

- records whose outcome is exactly `support` or `reject`;
- English civil disputes;
- records with a nonempty plaintiff claim and fact description;
- records within the configured input-length limit;
- records passing the explicit-outcome leakage screen.

Civil filtering excludes HKSAR prosecutions and lawsuit types or claims that
clearly concern criminal charges, sentencing, bail, or conviction.

The leakage screen rejects fact descriptions containing explicit holdings,
orders, dismissed or allowed claims, liability conclusions, damages awards,
credibility findings, or close textual reuse of the judgment-decision field.
Each rejected row receives one or more machine-readable exclusion reasons.

This produces a **low-explicit-leakage LegalHK subset**, not a claim that the
facts are genuinely independent of the outcome. LegalHK facts were augmented
using the source judgment, court reasoning, and decision. That dataset-level
limitation remains in every report.

Selection is deterministic under seed `20260619`. The evaluation set is
balanced 32/32 by outcome and distributed across issue count, input length,
defense presence, and lawsuit type. Smoke and evaluation IDs are disjoint.

Preparation writes:

- normalized selected cases;
- a manifest with selected counts and exclusion-reason counts;
- a condition-blind review file containing IDs, claims, facts, split, and
  selection metadata, but no gold outcomes, court reasoning, or decisions.

## Information boundaries

The four principal conditions receive:

- plaintiff claim and requested remedy;
- party names;
- numbered fact statements.

They do not receive:

- `related_laws`;
- reference issues;
- court reasoning;
- judgment decision;
- outcome labels.

The Oracle condition receives a sanitized reference state constructed from
LegalHK reference issues and related laws. It never receives court reasoning,
judgment prose, or the support/reject label. Oracle remains an upper-bound
diagnostic rather than a deployable method.

## Conditions

- **Direct:** one call producing support/reject and a concise rationale.
- **Structured:** one call producing issue conclusions, fact links, and the
  final support/reject decision.
- **Typed:** one call constructs a case state and a second call reasons from it.
- **Validated:** Typed plus deterministic state validation and at most one
  repair call.
- **Oracle:** analysis from the sanitized reference-derived state.
- **Sampling control:** three temperature-0.7 Structured outputs on 12 cases.

All output schemas are LegalHK-specific. `task_answer` and OpenExempt
non-binary answer handling are removed from the new pipeline.

## Metrics

Primary metrics:

- intention-to-treat support/reject accuracy;
- macro-F1 over support and reject;
- support recall and reject recall;
- structured-output failure rate.

Paired comparisons:

- condition accuracy difference on the same cases;
- paired bootstrap 95% confidence interval;
- exact McNemar test and discordant-case counts.

Reasoning and reliability metrics:

- conclusion-with-cited-fact rate;
- valid fact-ID rate and nonexistent-ID count as secondary hygiene measures;
- issue coverage against sanitized reference issues;
- decision/issue consistency;
- calls, tokens, latency, and failures;
- blinded local audit scores for issue coverage, rule fit, factual grounding,
  defenses, burden correctness, and final consistency.

Malformed, mixed, unresolved, or missing decisions count as incorrect in
intention-to-treat outcome metrics.

## Configuration and artifact isolation

Add `configs/legalhk_only.yaml`. It retains the existing model settings:

- `qwen3.5:9b`;
- context length 16,384;
- temperature 0;
- seed 20260619;
- state output limit 1,200;
- analysis output limit 1,000;
- concurrency 1.

The configuration uses:

- `data/processed/legalhk_only/`;
- `runs/legalhk_only/`;
- `reports/legalhk_only/`.

The raw LegalHK parquet cache remains shared under `data/raw/legalhk/`.
Existing mixed-pilot outputs under `runs/pilot`, `runs/smoke`, and
`reports/generated` are not modified.

## Acceptance criteria for this implementation

- All automated tests pass.
- `prepare --config configs/legalhk_only.yaml` selects five smoke and 64
  evaluation cases with no overlap and a 32/32 evaluation balance.
- Every selected case passes the deterministic explicit-leakage screen.
- `smoke --dry-run` reports 20 jobs.
- `generate --dry-run` reports 316 jobs.
- The actual 20-job smoke run completes without manual intervention.
- All four conditions return parseable outputs for all five smoke cases.
- Smoke scoring and resume checks succeed.
- The new main run is not started.

## Known limitations

- Automated and manual screening cannot remove latent outcome conditioning
  introduced during LegalHK dataset construction.
- Sixty-four evaluation cases can identify large effects and engineering
  failures but cannot establish a precise small effect.
- A LegalHK-only result does not establish cross-jurisdiction or cross-dataset
  generalization.
- The public LegalHK release has an unknown license, so selected case text and
  generated review files remain local and must not be redistributed.
