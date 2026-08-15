# Weekly Progress Report — 8–14 August 2026

## Results snapshot

All results below use the full LegalHK trajectory-development/validation split (n = 2,755). All four runs produced valid binary predictions for all records.

| Condition | Valid outputs | Accuracy | Weighted F1 | Mean output tokens per case |
| --- | --- | ---: | ---: | ---: |
| Direct | 2,755 / 2,755 (100%) | 75.25% | 75.47% | 108.8 |
| Structured IRAC | 2,755 / 2,755 (100%) | 71.69% | 72.07% | 215.8 |
| LegalFlux: no training | 2,755 / 2,755 (100%) | 65.74% | 66.18% | 1,928.3 |
| LegalFlux: SFT (`lr=2e-4`, epoch 6) | 2,755 / 2,755 (100%) | 65.15% | 64.71% | 1,173.2 |

The nine-checkpoint SFT grid (three learning rates × epochs 2, 4, and 6) is complete: all 72 shards finished, and each configuration produced 2,755/2,755 valid records. The best validation accuracy was 65.15% at `lr=2e-4`, epoch 6. In the paired comparison with the no-training run, both systems were correct on 1,370 cases; no-training alone was correct on 441; SFT alone on 425; and both were wrong on 519. The 0.58 percentage-point accuracy difference is small (paired McNemar p≈0.61), while SFT reduced output tokens by 39.2% and calls from 4.71 to 4.15 per case.

## What was completed this week

- Stabilized the minimal executor contract: each call follows one retrieved template and returns one concise instantiated result; executor artifacts are bounded to 180 words/1,800 characters with deterministic post-parse truncation as a last guard.
- Retained schema-constrained generation, resumable sharded ledgers, canary validation, a 32-case real-pipeline smoke test, and the forced-finalization retry for invalid reviewer decisions.
- Completed the full no-training LegalFlux validation run with 2,755/2,755 valid outputs.
- Completed and scored all nine structure-SFT checkpoints. The SFT checkpoint serves as both planner and reviewer; the base model remains the executor.
- Confirmed that our present SFT objective corresponds to ReasonFlux structure SFT: predict template description/scope from template name/tags. Trajectory preference learning/DPO remains the next training stage rather than part of this first SFT.

## LegalHK label semantics audit

The full local LegalHK file contains 18,374 cases: 7,030 `support` and 11,344 `reject`. The original LegalHK SFT instruction says to “respond to the plaintiff claim with support or reject,” so the primary target is the supplied claim—not a procedural keyword in isolation.

Appeal wording is usually consistent when the supplied claim is itself an appeal: among unambiguous surface matches, 126/146 allowed appeals are labeled `support`, while 501/504 dismissed appeals are labeled `reject`. However, some records describe an underlying plaintiff claim while an opposing party brings the appeal. For example, `legalhk-91` is labeled `support` when the defendant’s appeal is dismissed and the plaintiff’s underlying position is preserved; `legalhk-58` is labeled `reject` when a defendant’s appeal succeeds and the plaintiff’s default judgment is set aside. The audit also found apparent label/field inconsistencies, such as `legalhk-806` and `legalhk-673`, where the recorded plaintiff succeeds but the label is `reject`. We should treat this as dataset noise rather than invent a universal “appeal dismissed = reject” rule.

Recommended reviewer instruction:

> When `decision` is `final_answer`, `final_decision` must be exactly `support` or `reject`. Interpret the label relative to the supplied CLAIM OR TASK, not merely the procedural posture or the word “appeal.” `support` means that the court grants, allows, upholds, or preserves the claim or requested relief stated there. `reject` means that the court dismisses, refuses, or denies it. If the CLAIM OR TASK is itself an appeal, `support` means the appeal is allowed and `reject` means it is dismissed. If an opposing party appeals an underlying claim, classify whether the supplied claim or relief succeeds. Make `final_decision` agree with `final_rationale`.

## Comparative error case studies

### Case 1 — `legalhk-10140`: correct rationale, inverted SFT decision

Claim: declarations and injunctions for trademark infringement and passing off. Gold: `support`.

Key facts: the writ and summons were served; the defendant filed no acknowledgement or defence and did not appear; the plaintiff abandoned declarations but retained injunction claims; the pleadings allege use of the plaintiff’s mark and a confusing domain in the same trade; Order 19 rule 7 requires judgment from the pleadings alone.

No-training trace:

1. Planner proposed four steps: default-judgment threshold; Order 19 rule 7 evidence restriction; passing-off/trademark elements; standing and mark validity.
2. Retrieval selected LF008, “Setting Aside Default Judgment for Irregularity of Service,” by tag-overlap embedding (0.811). The template direction was imperfect, but the executor recognized that service was regular and that the case should proceed to the injunction merits.
3. The executor concluded that the procedural threshold was met and that the pleadings supported the injunction.
4. The reviewer finalized after one execution: its rationale said the injunction should be granted and returned `support`.

SFT trace:

1. Planner proposed three steps: enter default judgment; assess pleadings for injunction; formulate injunction terms.
2. Retrieval again selected LF008, now through `exact_tag_unique` with an artificial score of 1.0 because `default_judgment`/`service_of_process` left one tag candidate.
3. The executor nevertheless concluded that the court enters default judgment on the remaining injunction claims.
4. The reviewer’s analysis and final rationale both said judgment/injunctions should be granted, but `final_decision` was `reject`.

Diagnosis: this is a pure reviewer label–rationale alignment failure. Retrieval was directionally imperfect in both runs, yet the reasoning remained sufficient. The proposed final-decision instruction and a contradiction-triggered retry directly target this error.

### Case 2 — `legalhk-10034`: retrieval mismatch plus premature finalization

Claim: recovery of HK$31,737,731.62 under a renovation contract. Gold: `reject`.

Key facts: the contractor completed the work, but the contract was procured through agreed tender-rigging and payments totaling 17.5% of the contract sum; participants were convicted of bribery/conspiracy; the conviction materials were admissible to identify the underlying facts.

No-training trace:

1. Planner proposed four steps: prove the bribery conspiracy; admit prior convictions despite Hollington; apply illegality/ex turpi causa; consider restitution or discretionary relief.
2. Retrieval selected LF143 (settlement-offer evidence; mismatch, 0.598), then LF078 (Hollington/prior convictions; exact-tag match), then LF168 (illegality in contract enforcement; 0.707).
3. Executors respectively established the conspiracy, admitted the conviction evidence, and concluded that the illegally procured contract was unenforceable.
4. Reviewers continued through the decisive evidence and illegality steps, then stopped before the unnecessary restitution step and returned `reject`.

SFT trace:

1. Planner also recognized four issues: contract balance; bribery evidence; prior-conviction evidence; public-policy apportionment.
2. Retrieval selected LF041, “Land Resumption Compensation Rules” (full-pool embedding 0.622), then LF173, “Duress as a Defense to Written Debt Acknowledgments” (0.663). Neither matched the planned legal task.
3. The first executor merely restated the balance. The second used the duress template to reframe bribery as separate from debt validity and concluded that the contract remained enforceable.
4. The first review correctly said the bribery/public-policy steps remained necessary. After the second execution, the reviewer abandoned those planned steps, finalized early, and returned `support`.

Diagnosis: the template pool already contains strong relevant templates (LF078 and LF168), so pool quality is not the sole cause. The live retriever selected mediocre top-1 neighbors without reranking, and the reviewer then treated a template-induced detour as dispositive.

### Case 3 — `legalhk-1012`: retrieval mismatch survives, but SFT label still flips

Claim: Norwich Pharmacal pre-action discovery of bank records for funds transferred by fraud. Gold: `support`.

Key facts: HK$2.436 million was transferred without the plaintiff’s knowledge; the bank held the target account and required a court order; the plaintiff supplied cogent fraud evidence, exhausted other avenues, and requested only necessary records.

No-training trace:

1. Planner proposed four steps: neutral-bank instrument status; necessity/proportionality; the court-order objection; statutory discovery power.
2. Retrieval produced three imperfect templates—LF030 (procedural default), LF049 (proceeds-of-crime forfeiture), and LF024 (reopening judicial review)—but the executors stayed grounded in the case and applied the Norwich Pharmacal test.
3. The reviewer continued through the necessity and procedural-objection analysis and returned `support` with a rationale that the narrow disclosure order should be granted.

SFT trace:

1. Planner proposed Norwich Pharmacal identification, threshold, and scope steps.
2. Retrieval selected LF023, “Interpleader Relief for Neutral Financial Stakeholders” (full-pool embedding 0.599).
3. The executor correctly applied Norwich Pharmacal and concluded that the bank should disclose the records.
4. The reviewer immediately finalized; its rationale said “the court should grant the application,” but `final_decision` was `reject`.

Diagnosis: this repeats the label–rationale contradiction even when the executor reaches the correct legal result, showing that retrieval improvements alone will not fix final accuracy.

## Retrieval pipeline assessment

Current live retrieval is: exact normalized name → one exact tag candidate (returned immediately with score 1.0) → BGE-M3 dense top-1 over the tag-overlap subset → BGE-M3 dense top-1 over the full pool. The query contains only the planner’s step name, description, and tags; it omits the case claim, facts, authorities, and procedural posture. The live path has no cross-encoder or LLM reranking. This explains why generic tags can force LF008 and why full-pool similarities around 0.60–0.66 can select LF041/LF173/LF023.

Always using full-pool semantic top-1 would remove the brittle tag gate but would not solve these mediocre-neighbor errors. The better design is high-recall retrieval followed by strong selection:

1. Treat exact name/tag matches as ranking features, not automatic winners; require a semantic score/margin even for a unique tag match.
2. Build the retrieval query from the planned step plus the claim, key facts, authorities, and lawsuit/procedural type.
3. Retrieve top-k (approximately 8–12) from the entire 227-template pool with a hybrid sparse+dense method, then rerank with `BAAI/bge-reranker-v2-m3`.
4. Add an agentic selector only after shortlisting. Show the selector candidate cards containing template name, tags, description, and application scenario; let it choose one, request a different shortlist, or abstain. Feeding all 227 full templates at once would add context and menu-selection noise; a compact all-name/tag catalog can be used only for coarse routing.
5. Create an explicit retrieval benchmark: manually label acceptable templates for a representative set of planned steps, then measure Recall@k, MRR, and step coverage independently of executor/reviewer accuracy.
6. Run a controlled ablation with cached plans: current retriever; full-pool BGE-M3 top-1; hybrid top-k + cross-encoder; hybrid top-k + LLM selector. This isolates retrieval from planner variation.

## Recommended next steps

1. Add the target-relative reviewer instruction above to both ordinary review and forced-finalization prompts. Add a narrow retry when the rationale explicitly says grant/allow while the label is `reject`, or dismiss/deny while the label is `support`; log every retry rather than silently rewriting the answer.
2. Replace `exact_tag_unique` as an unconditional shortcut and implement the hybrid top-k + reranker baseline before trying a fully agentic menu.
3. Add case context to the retrieval query and build the retrieval-only evaluation set.
4. Re-run a small fixed-case ablation first, then the full validation set only for the best retrieval variant.
5. Because direct remains the strongest baseline, treat LegalFlux’s current value as traceability/controllability and use ablations to establish where accuracy is lost before DPO.
6. Construct DPO trajectories only after reviewer label alignment and retrieval reliability stabilize; otherwise preferences will reward pipeline artifacts rather than planning quality.
7. After stabilizing the pipeline, rerun direct, structured, and no-training LegalFlux under the same vLLM 0.21.0 setup for an apples-to-apples engine comparison.

## Runtime note: vLLM 0.21.0

The earlier vLLM 0.18.1 serving path repeatedly failed while loading the Qwen3.5 LoRA checkpoints, including errors caused by vision-module adapter targets; `--enforce-eager` alone did not resolve adapter compatibility. We therefore pinned vLLM 0.21.0, prepared text-only serving adapters, and kept eager execution. This combination passed the canary and supported the completed 72-shard SFT grid. The engine change was an operational compatibility fix, not an intended model-quality intervention, so older direct/structured/no-training results should eventually be rerun under 0.21.0 before making finely grained comparisons.
