# Motivation Memo: LegalHK Cases Exhibit Heterogeneous Reasoning Trajectories

This memo uses the raw LegalHK `court_reasoning` field to identify dataset-internal evidence for motivating LegalFlux. The goal is not to claim that the current pipeline already solves the task, but to justify why a fixed one-size-fits-all reasoning scaffold is mismatched to the dataset.

## Aggregate Signals

Raw source: `data/raw/legalhk/train.parquet`

- Rows inspected: 18,374
- Rows with non-empty `court_reasoning`: 18,373
- Rows with supplied `related_laws`: 14,324
- Rows with supplied `relevant_cases`: 13,468
- Rows with estimated multiple issues in the `issues` field: 15,339

Regex audit over `court_reasoning` suggests that the dataset mixes several reasoning families:

- `precedent_analogy`: 9,072 rows
- `remedy_or_discretion`: 8,192 rows
- `statutory_or_rule_application`: 4,963 rows
- `multi_issue_composition`: 4,549 rows
- `evidence_credibility_burden`: 3,542 rows
- `procedural_gateway`: 3,250 rows
- `injunction_discretion`: 729 rows

These counts are only heuristic, but they show that LegalHK is not homogeneous. Many cases require identifying the procedural posture, remedial standard, authority type, evidence burden, or issue structure before deciding whether the plaintiff's claim is supported.

## Curated Examples

### 1. Summary Judgment: Procedural Gateway Before Merits

Case: `legalhk-506`

Lawsuit type: `Order 14 summary judgment`

Gold label: `reject`

Claim: the plaintiff claimed unpaid monthly service fees under a contract.

Dataset reasoning excerpt:

> "The principles on Order 14 are trite: the defendant must show that there are triable issues."

> "the relevant test is whether the defendant has raised credible triable issues."

> "The court will not take the alleged defence on its face value but test it against the evidence disclosed"

Implied trajectory:

1. Identify procedural posture: summary judgment.
2. Retrieve procedural threshold: whether there is a real/credible triable defence.
3. Test the defence against affidavits and contemporaneous documents.
4. Decide whether the plaintiff should get judgment now.

Why this matters for LegalFlux:

This is not a generic merits-only contract analysis. A fixed IRAC prompt may over-focus on whether the service fees were unpaid, while the court's actual path is threshold-based: whether the defendant has raised a believable defence sufficient to avoid summary judgment.

### 2. Mareva Injunction: Remedy-Specific Multi-Factor Discretion

Case: `legalhk-46`

Lawsuit type: `Mareva injunction application`

Gold label: `support`

Claim: the plaintiff sought to freeze assets in Hong Kong for alleged unjust enrichment.

Dataset reasoning excerpt:

> "four factors to grant a Mareva injunction: (1) a good arguable case ... (2) assets within the jurisdiction ... (3) the balance of convenience ... and (4) a real risk of dissipation of assets."

> "The balance of convenience is in favour of granting the injunction"

Implied trajectory:

1. Recognize remedy type: Mareva/freezing injunction.
2. Apply the injunction-specific checklist.
3. Check arguable substantive claim.
4. Check jurisdictional assets.
5. Check risk of dissipation and balance of convenience.
6. Decide whether interim relief is justified.

Why this matters for LegalFlux:

This case needs a remedial-discretion template rather than a simple liability template. The final answer depends on whether several injunction prerequisites are satisfied, not only on whether unjust enrichment might eventually be proved.

### 3. Statutory Interpretation: Meaning, Purpose, and Distinguishing Authorities

Case: `legalhk-5`

Lawsuit type: `Summons to strike out claim for lack of any reasonable cause of action`

Gold label: `reject`

Claim: recovery of possession for self-occupation.

Dataset reasoning excerpt:

> "examined the meaning of the terms 'a landlord' and 'the landlord' in Section 36"

> "considered the purpose of Section 53(2)(b) and Section 36"

> "distinguishing the cases of Cheung Hei v. Yung Yee-kam and Loke Choong-wing v. Lai Lok-sin"

Implied trajectory:

1. Identify statutory phrase whose meaning controls the claim.
2. Compare related statutory provisions.
3. Infer statutory purpose.
4. Distinguish precedent.
5. Apply the interpretation to plaintiff's status as successor-in-title.

Why this matters for LegalFlux:

This case is driven by interpretive reasoning, not fact weighing. A good trajectory must foreground statutory meaning and precedent distinction before deciding whether the plaintiff can invoke the eviction right.

### 4. Evidential Credibility: Fact Reconstruction Without Supplied Law or Cases

Case: `legalhk-6355`

Lawsuit type: blank

Gold label: `support`

Claim: recovery of the equivalent of a paid sum of GBP 1,000.

Dataset reasoning excerpt:

> "determine the purpose for which the GBP 1,000 was sent"

> "consider the credibility of the parties and their versions of events"

> "consider the documentary evidence, including the correspondence between the parties"

Implied trajectory:

1. Identify the central factual dispute: why the money was sent.
2. Compare competing narratives.
3. Evaluate party credibility.
4. Use documentary correspondence to infer purpose.
5. Decide whether the defendant was entitled to retain/use the money.

Why this matters for LegalFlux:

This is almost the opposite of the statutory-interpretation case. The useful high-level template is evidence and credibility assessment, not rule extraction or precedent analysis.

### 5. Stay of Civil Proceedings: Precedent-Guided Balancing

Case: `legalhk-9741`

Lawsuit type: `application to stay proceedings`

Gold label: `reject`

Claim: fraudulent misappropriation of more than HKD 130 million.

Dataset reasoning excerpt:

> "applied the principles set out in Jefferson v Bhetcha"

> "a balancing exercise ... between the defendant's right to remain silent and the plaintiffs' right to have their claim processed and heard"

> "no real danger of the causing of injustice"

Implied trajectory:

1. Recognize procedural remedy: stay of civil proceedings due to concurrent criminal proceedings.
2. Retrieve precedent-guided balancing test.
3. Weigh defendant's silence/fair-trial interest against plaintiff's right to proceed.
4. Assess whether injustice is real rather than speculative.
5. Decide whether to stay the civil claim.

Why this matters for LegalFlux:

This case is neither ordinary fraud merits reasoning nor pure procedure. It needs a specific balancing template keyed to concurrent civil/criminal proceedings and precedent.

### 6. Public Law / Legitimate Expectation: Multi-Issue Decomposition

Case: `legalhk-13093`

Lawsuit type: `judicial review`

Gold label: `reject`

Claim: applicants argued they were beneficiaries of Court of Final Appeal judgments and had legitimate expectations.

Dataset reasoning excerpt:

> "not beneficiaries ... as they were not parties to those proceedings"

> "did not amount to a clear and unambiguous representation"

> "the Director's interpretation of the Concession was reasonable"

> "exercised his discretion reasonably"

Implied trajectory:

1. Decompose the case into linked public-law issues.
2. Decide whether prior judgments cover the applicants.
3. Apply the legitimate-expectation representation requirement.
4. Review administrative interpretation of the concession.
5. Review discretionary removal orders for reasonableness.
6. Compose issue outcomes into the final judgment.

Why this matters for LegalFlux:

This case requires a trajectory across multiple public-law subquestions. A single fixed template risks flattening the reasoning into one generic rule-application step, while the court's reasoning moves through coverage, representation, interpretation, and discretion.

## Suggested Paper Motivation Language

LegalHK itself provides evidence that legal judgment prediction is not only a matter of producing more reasoning, but of selecting the right kind of reasoning. In one row, the court frames the task as an Order 14 summary-judgment threshold and asks whether the defendant raised "credible triable issues"; in another, a Mareva injunction case turns on "four factors" including jurisdictional assets, balance of convenience, and risk of dissipation; in another, the court's reasoning centers on the "meaning" and "purpose" of statutory provisions; and in a fact-heavy debt dispute, the court instead asks why money was transferred and evaluates "the credibility of the parties" and documentary correspondence. These examples show that LegalHK cases encode heterogeneous reasoning trajectories. Therefore, a method such as LegalFlux is motivated not simply by the desire to add more steps, but by the need to adapt the high-level reasoning path to the case's legal posture, authority structure, evidential burden, and remedial form.

## Caveat

LegalHK's released fields appear to be distilled summaries rather than full original judgments. The examples above should therefore be presented as evidence of heterogeneity in the dataset's reasoning annotations, not as proof that the released input contains every fact needed to reproduce full judicial reasoning.
