You are consolidating candidate LegalFlux templates into one compact,
high-quality library for Indian Supreme Court cases.

Review the complete candidate set together. Keep only reusable, distinct legal
reasoning operations at a middle-to-high abstraction level. There is no required
or preferred final template count: retain every candidate that independently
meets the quality standard, and do not add or remove templates to reach a quota.

Merge candidates that perform substantially the same reasoning operation.
Remove candidates that are overly broad, overly specific, directionally tied to
an outcome, weakly supported, wholly subsumed by another template, or primarily
summaries of substantive law rather than executable reasoning procedures.

Treat consolidation as selective preservation rather than wholesale rewriting.
When retaining one candidate without merging it, copy all of its executable
template fields verbatim. Rewrite executable fields only when merging multiple
candidates; then change only what is necessary to produce one coherent template.
Do not paraphrase a retained candidate merely for style.

For each retained template:

- use a concise retrieval-friendly name and normalized tags;
- state clear positive application conditions and meaningful boundaries;
- provide an ordered operational reasoning flow;
- use a synthetic example that does not reproduce a source case;
- record every source_candidate_id that was retained or merged into it;
- do not copy source case IDs, parties, dates, amounts, citations, outcomes, or
  F-numbered facts into the executable template fields.

Return one JSON object matching the consolidation output schema. Template IDs
will be assigned deterministically after this consolidation call.
