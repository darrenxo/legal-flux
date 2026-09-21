# LegalHK Issue-Graph and Legal-Operator Pilot

## Purpose

This pilot tests a compact alternative to free-form LegalFlux trajectories:

1. Construct a shallow issue dependency graph from the supplied case only.
2. Retrieve one typed, high-level legal operator for each issue.
3. Resolve child issues in separate calls before their parent issue.
4. Synthesize the final `support` or `reject` prediction from the resolved
   top-level issue.

The graph topology is inspired by LEGIT's issue-tree representation, especially
its separation of top-down issue decomposition from bottom-up deduction. This
pipeline does **not** reproduce LEGIT's gold-rubric construction: LEGIT builds
rubrics from completed judgments, while this pilot constructs its graph at
inference time without court reasoning, judgment decisions, gold labels,
reference issues, or reference states.

Primary reference: Jinu Lee et al., "Evaluating Legal Reasoning Traces with
Legal Issue Tree Rubrics," ACL 2026 ([paper](https://aclanthology.org/2026.acl-long.150/),
[repository](https://github.com/jinulee-v/LEGIT)).

## Minimal graph

- The supplied plaintiff's claim is the immutable `ROOT`.
- `I1` is the single top-level adjudicative question and must preserve the
  actual procedural posture.
- Up to three additional nodes provide the legal, factual, defensive, or
  remedial premises needed to resolve their parent.
- The initial pilot permits at most four issues and depth two.
- The graph planner cannot output issue conclusions or a final label.

Issue executors receive only the raw facts and authorities assigned to their
node, plus already-completed child findings. The top-level finding is resolved
last. A resolved top-level conclusion deterministically constrains the final
binary label, preventing rationale-label reversal.

## Operator library

`templates/legal_operators_v0.jsonl` contains 12 operations:

1. Procedural Gate Check
2. Claim or Issue Element Decomposition
3. Governing Text Construction
4. Precedent Comparison and Authority Synthesis
5. Burden, Presumption, and Standard of Proof
6. Evidence Reliability and Fact Finding
7. Defense, Exception, or Legal Bar Test
8. Causation and Scope of Responsibility
9. Loss and Quantum Assessment
10. Discretionary Balancing and Proportionality
11. Appellate or Supervisory Error Review
12. Remedy Availability and Scope

These are reusable reasoning operations rather than case-derived mini-opinions.
Their design was informed by an audit restricted to the 12,859 LegalHK cases in
`template_source` and `planner_train`, plus general legal reasoning structure.
The manifest at `templates/legal_operators_v0.manifest.json` records this
provenance. No `trajectory_dev` or `final_test` gold reasoning was inspected to
construct the library.

Retrieval first filters by the graph planner's typed `issue_type`, then ranks
only compatible operators with BGE-M3 embeddings. This makes the type decision
explicit and keeps semantic retrieval narrow.

## Local pilot result

Model: local `qwen3.5:9b`, hidden thinking disabled by the existing Ollama
configuration, temperature 0, fixed project seed. Evaluation cases are the
first eight fixed `trajectory_dev` records.

| Condition | Valid | Accuracy | Weighted F1 | Mean calls | Mean input tokens | Mean output tokens |
|---|---:|---:|---:|---:|---:|---:|
| Matched direct prompt | 8/8 | 75.0% | 75.0% | 1 | existing run | existing run |
| Initial flat issue graph | 8/8 | 50.0% | 43.33% | 6.0 | 13,901.6 | 2,588.5 |
| Rooted, evidence-scoped graph | 8/8 | 50.0% | 43.33% | 6.0 | 14,929.4 | 2,704.8 |

The sample is too small for a performance claim. The important pilot findings
are:

- Schema reliability is good: both graph conditions produced 8/8 valid records
  with no generation failures.
- The rooted design creates inspectable dependencies and eliminates final-label
  reversal when `I1` is resolved.
- The tested predictions remain strongly support-biased (seven `support`, one
  `reject`) and do not beat the matched direct baseline.
- Better trace structure alone did not improve accuracy on this sample, while
  substantially increasing calls and tokens.

Therefore this pilot supports the pipeline as an analysis instrument, but it
does not yet support scaling the method as a performance experiment. The next
methodological question is how to calibrate the top-level claim-relative issue
resolution, especially leave, appeal, and judicial-review postures, without
using validation labels or gold reasoning.

## Commands

```powershell
$env:PYTHONUTF8 = "1"

.\.venv-codex\Scripts\python.exe -m legal_pilot `
  --config configs\legal_flux.yaml `
  flux-operator-pilot `
  --phase trajectory-dev `
  --case-limit 8 `
  --run-tag issue-graph-operator-rooted-scoped-8-v2 `
  --fail-on-errors
```

Artifacts are written to:

`runs/legal_flux/operator_pilot/trajectory_dev/experiments/<run-tag>/`

The ledger records the issue graph, selected operator per issue, each issue
finding, final decision, token/call counts, prompt hashes, repairs, and schema
errors.
