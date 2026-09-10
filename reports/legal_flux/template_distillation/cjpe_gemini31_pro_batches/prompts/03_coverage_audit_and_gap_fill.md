You are auditing one original source-case batch against the current
consolidated LegalFlux template library.

Inspect every supplied case's full_text, including the facts, procedural history, arguments, authorities, lower-court rulings, and present-court reasoning or judgment it contains. Identify whether the library covers the recurring legal reasoning
operations actually exhibited by this batch. An individual uncovered case is
not a library gap. Propose a gap candidate only when the same missing operation
is supported by the configured minimum number of supplied cases and satisfies
the same abstraction, reuse, manifestation, and boundary requirements as the
initial candidate stage.

Do not restate a legal topic, reproduce a source outcome, or propose a candidate
already covered or subsumed by the current library. Zero gap candidates is
valid and preferable to a weak addition. Gap candidates are proposals only and
will undergo a separate global adjudication before entering the final library.

Return one JSON object matching the gap-audit output schema.

Aggregate source coverage metadata:

```json
{
  "template_source_cases": 994,
  "coarse_legal_family_counts": {
    "other_uncertain": 994
  },
  "coarse_legal_family_batch_counts": {
    "other_uncertain": 33
  },
  "primary_family_counts": {
    "general_legal_reasoning": 994
  },
  "demand_focus_counts": {
    "general_resolution": 994
  },
  "all_reasoning_demand_counts": {},
  "trajectory_prefix_counts_top50": {
    "unknown": 994
  },
  "batch_count": 33,
  "batched_case_ids": 994,
  "batch_kind_counts": {
    "semantic_family": 33
  }
}
```
