You are constructing a compact, high-quality library of reusable legal
reasoning templates for Indian Supreme Court cases.

Each supplied source case provides its complete case text in one
`full_text` field. Infer reusable legal reasoning operations from that text,
which may include facts, procedural history, party arguments, authorities,
lower-court rulings, and the present court's reasoning or judgment. Treat
explicit or implicit outcome language as evidence of the court's reasoning
process, never as a template target or prediction shortcut.

Analyze the supplied batch and return at most 5 candidate
templates. Zero candidates is valid and preferable to creating a weak,
duplicative, overly broad, or overly specific template.

A candidate is eligible only when:

1. At least 3 supplied cases exhibit the same underlying
   legal reasoning operation.
2. The supporting cases include at least two meaningfully different factual or
   procedural manifestations of that operation.
3. The template would provide useful guidance for unseen cases of similar type.
4. Its application can be described with clear positive triggers and meaningful
   boundaries.
5. Its reasoning flow performs a distinct legal operation rather than merely
   restating a legal topic.

TARGET ABSTRACTION:

A good template is a legal reasoning operation with a middle-to-high abstraction
level, such as a procedural gate, allocation of burden, structured evidential
assessment, authority-synthesis method, multi-factor legal test, defense
analysis, remedy selection, or appellate review operation.

Reject a candidate if it is:

- so general that it merely says to identify issues, apply law to facts,
  consider evidence, or reach a conclusion;
- tied to one case, one unusual fact pattern, one party, one outcome, or a
  narrowly worded procedural event;
- primarily a summary of substantive law rather than an executable reasoning
  procedure;
- directionally framed to produce a particular support/reject result;
- substantially equivalent to another candidate in this batch;
- wholly contained within another candidate that performs the same reasoning
  operation.

For every candidate:

- Use a concise, retrieval-friendly template_name.
- Use a concise set of normalized knowledge_tags.
- Make description explain the reusable reasoning operation.
- Make application_scenario state when to use it and, if applicable and helpful,
  when not to use it.
- Give at least 2 ordered, operational reasoning_flow steps.
- Use a synthetic example_application that does not reproduce a source case.
- Record the supporting_case_ids and explain the shared_pattern.
- Make scope_exclusions identify contexts where the operation should not be used.
- Make support_count equal the number of distinct supporting_case_ids.
- Do not copy party names, dates, amounts, citations, outcomes, or F-numbered
  facts.

Before returning, compare all proposed candidates with one another. Merge
near-duplicates and remove any candidate subsumed by another.

Return one JSON object matching the candidate-template output schema.
