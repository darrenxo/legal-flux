# Weekly Research Report

## Executive summary

- Completed the full Qwen3.5-9B BF16 direct-versus-structured evaluation on AnnoCaseLaw, Realistic LJP Facts, and IL-TUR/CJPE: 6,840 generations with zero errors.
- The one-shot structured IRAC prompt did not reliably improve accuracy. Its change relative to direct was +0.5 percentage points on AnnoCaseLaw, -0.7 on Realistic LJP Facts, and -0.5 on IL-TUR/CJPE; none of the paired differences was statistically significant.
- The refined SFT checkpoint search selected learning rate 5e-5 at epoch 4 as the best planner checkpoint. This checkpoint is now the source policy and reference adapter for trajectory DPO.
- DPO preference-data construction is in progress. We sample four trajectories per planner-training anchor, execute each fixed trajectory on the anchor and its two X_sim neighbors, use mean outcome accuracy as the reward, and export the highest-versus-lowest scoring trajectory as a preference pair before DPO training.
- Dataset and rationale inspection supports the template-trajectory motivation: several higher-level operations recur across cases, but the relevant subset and ordering are case-dependent. A universal IRAC scaffold helps some cases and harms others.

## 1. Direct and structured benchmark results

| Dataset | Majority | Direct | Structured | Difference | Paired p |
| --- | ---: | ---: | ---: | ---: | ---: |
| AnnoCaseLaw | 46.95% | 50.00% | 50.51% | +0.51 pp | .911 |
| IL-TUR/CJPE | 50.23% | 70.20% | 69.74% | −0.46 pp | .592 |
| Realistic_LJP_Facts | 50.30% | 52.55% | 51.89% | −0.66 pp | .308 |

The full evaluation used Qwen3.5-9B in BF16 on Delta. AnnoCaseLaw contains 394 facts-plus-procedural-history cases with three labels (affirm, reverse, mixed). Realistic LJP Facts contains 1,509 facts-only Indian Supreme Court test cases, and IL-TUR/CJPE contains 1,517 outcome-redacted full case documents.

- AnnoCaseLaw: direct 50.0%; structured 50.5%; majority baseline 47.0%; macro-F1 0.344 versus 0.298.
- Realistic LJP Facts: direct 52.6%; structured 51.9%; majority baseline 50.3%; macro-F1 0.427 versus 0.405.
- IL-TUR/CJPE: direct 70.2%; structured 69.7%; majority baseline 50.2%; macro-F1 0.700 versus 0.693.

Paired McNemar tests give p = .911, .308, and .592 for AnnoCaseLaw, Realistic LJP Facts, and IL-TUR/CJPE, respectively. The generic structured prompt therefore adds output length and latency without a reliable accuracy gain. This is an important negative result: more visible structure is not automatically better reasoning.

These are paired views of the same ILDCmulti judgments rather than independent case collections. All 1,509 Realistic LJP test cases match 1,509 of the 1,517 CJPE test cases, with identical labels; CJPE has eight additional cases. Realistic LJP Facts keeps only sentences automatically assigned the rhetorical role “facts.” CJPE keeps the cleaned, anonymized proceedings—including lower-court rulings, arguments, statutes, precedents, and much of the present court's analysis—but deletes the ending section(s) that directly state the final decision and extracts the label from those deleted sections. Across the 1,509 matched cases, the median input is 3,590 characters for facts-only and 17,391 for CJPE, a 4.64× median length ratio.

The approximately 17.7-point direct-accuracy gap (52.6% versus 70.2%) therefore measures the value of the additional procedural and legal context, not simply a difference in case difficulty. It still does not prove deeper reasoning: the CJPE paper reports that the final portions before the removed disposition contain especially strong ratio and outcome-adjacent cues, and 113 of our 1,517 CJPE inputs were truncated at the benchmark character limit.

## 2. Refined best SFT checkpoint

The refined checkpoint evaluation selected the template-structure SFT checkpoint trained with learning rate 5e-5 at epoch 4. Selection was performed on the development trajectory task rather than the sealed final test set. This is now the best SFT checkpoint used for downstream trajectory generation and as the initial/reference policy for DPO.

The practical implication is that we no longer need to compare coarse learning-rate endpoints for the current DPO run. The remaining question is whether preference learning can make the selected SFT planner choose better case-adaptive trajectories—not merely reproduce the template descriptions learned during SFT.

## 3. DPO progress and expected runtime

The DPO pipeline is running on the selected 5e-5, epoch-4 SFT checkpoint. The current bottleneck is preference-data construction, which precedes the comparatively smaller DPO optimization step.

- X_sim construction: for each of 9,185 planner-training anchors, retrieve two legally similar cases using BGE-M3 dense retrieval followed by cross-encoder reranking. Each evaluation set therefore contains the anchor plus two X_sim neighbors.
- Trajectory sampling: sample exactly four stochastic planner trajectories per anchor from the selected SFT checkpoint.
- Reward evaluation: retrieve the template sequence once for each candidate and execute that fixed sequence on all three X_sim cases with the unchanged base executor. The reward is mean binary prediction accuracy over those three cases.
- Pair construction: select the highest-scoring complete trajectory as chosen and the lowest-scoring complete trajectory as rejected. Skip anchors for which all complete candidates have the same accuracy.
- DPO training: export canonical chosen/rejected planner JSON and continue training the selected SFT adapter with DPO. Executor traces and final labels provide reward evidence; they are not used as assistant targets.

At full scale this means 36,740 sampled trajectories and as many as 110,220 fixed-trajectory case executions before finalization calls and retries. Even with sharding and four concurrent GPU jobs, data collection is expected to take days and may extend to weeks depending on queue time and per-case document length. The run is resumable, so completed candidates and evaluations are preserved across job restarts.

## 4. What counts as gold reasoning in these datasets?

AnnoCaseLaw provides the strongest direct supervision for our motivation. The source cases contain expert annotations for Facts, Procedural History, Relevant Precedents, Application of Law to Facts, and Outcome, plus negligence concepts. These are gold rationale spans extracted from the court opinion. They are not a canonical ordered trajectory, but they show which legal operations and evidence were material.

Across the 394 benchmark cases, every case has at least one Application-of-Law-to-Facts span and at least one Relevant-Precedent span. The median case has five application spans; 380 cases have at least two and 251 have at least five. Concept annotations align to 391 cases and contain 32 core concepts. The median case activates two concepts, 261 cases activate at least two, 145 activate at least three, and the corpus contains 127 distinct exact concept combinations. The most common exact combination appears in only 98 cases (25.1%). This is direct corpus-level evidence that reasoning operations recur but do not collapse into one universal sequence.

ILDC/CJPE provides a smaller but high-quality expert set. The official repository includes 56 test documents annotated independently by five legal experts, and the released ranked explanations are now available locally and align exactly to our CJPE and facts-only case IDs. Each expert predicted the decision and marked explanatory sentences. Rank 1 is the highest-priority tier: sentences immediately leading to the decision. Rank 2 contains contributing reasons, Rank 3 highlights disagreement with a lower court or tribunal, and Rank 4 or numerically larger ranks contain essential facts or increasingly indirect background. The ranks express importance, not the temporal order of a reasoning procedure. Experts may select multiple sentences at a rank and may disagree on both the selected evidence and the predicted verdict.

The released JSON contains nonempty annotations through Rank 10, but later tiers are sparse: across 280 expert–case assignments, 275 have Rank-1 text, whereas only three have Rank-9 and three have Rank-10 text. These are gold explanation/evidence annotations, not gold step-by-step trajectories. They can test whether a generated rationale covers decisive rules, facts, and procedural signals, but they do not specify a unique sequence of operations.

Realistic LJP Facts has no separate gold reasoning annotations, but every one of the 56 expert cases has both a CJPE full-proceedings view and a facts-only view. On this aligned expert subset, BF16 direct/structured accuracy is 67.9%/64.3% for CJPE and 53.6%/53.6% for facts-only. This enables controlled analysis of what information changes a prediction while using the same expert evidence as the reference.

## 5. Case studies: reusable steps, case-dependent trajectories

### AnnoCaseLaw 0001 — Cox v. May Department Store (gold: reverse)

The plaintiff's jacket became trapped in an escalator while she was riding normally. The trial court granted summary judgment because no specific defect or negligence was shown and it rejected res ipsa loquitur. The gold application spans separately evaluate: whether this type of accident ordinarily implies negligence; expert evidence; defendants' control over the escalator; the distinction between the escalator and the jacket as the relevant instrumentality; comparative negligence; and the summary-judgment threshold.

In the full BF16 run, both direct and structured rationales incorrectly affirmed. Both treated inspections by the city and maintenance provider as defeating exclusive control and over-weighted the absence of a detected defect, even though res ipsa loquitur exists precisely because a plaintiff may lack direct proof of the defect. The structured answer stated the doctrine's elements but still mapped the evidence to the wrong legal conclusion.

Motivated trajectory: procedural posture -> identify dispositive doctrine -> check each res ipsa element -> map evidence to each element -> apply the summary-judgment evidentiary threshold -> disposition. A generic IRAC heading is insufficient; this case needs a doctrine-specific element checklist.

### AnnoCaseLaw 0029 — Torres v. State (gold: reverse)

Police received detailed information about a suspected murderer, delayed acting, and failed to prevent his interstate flight and later killings. The lower courts treated limited public resources and rising crime as policy reasons not to recognize a duty. The gold application spans make a finer distinction: resource constraints may bear on breach, but not on the existence of the statutory investigative duty; foreseeability was for the factfinder; and the duty could extend to foreseeable victims outside the state.

Both BF16 conditions incorrectly affirmed by repeating the Court of Appeals' policy reasoning from the procedural-history section. They failed to recognize that the current court was reviewing—and rejecting—that intermediate holding.

Motivated trajectory: identify current tribunal and challenged holding -> separate duty, breach, foreseeability, and policy -> determine judge-versus-jury allocation -> apply statutory scope -> disposition. This differs materially from the element sequence needed in case 0001.

### CJPE 1962_47 (gold: rejected)

This case is a review petition following an earlier accepted appeal. The paper reports that two of five legal experts predicted acceptance because a sentence referred to the Supreme Court having accepted the earlier appeal. The present matter, however, was the later review petition; reaffirming the earlier judgment meant rejecting the current petition. In BF16, CJPE direct and structured both handled this correctly. In the paired facts-only view, direct made exactly the proceeding-confusion error and predicted acceptance from the earlier appeal, while structured identified the present review petition and correctly rejected it.

Motivated trajectory: identify the current proceeding -> separate prior disposition from present requested relief -> track which party is now seeking review -> evaluate the review grounds -> map the conclusion to the current petition's label. This is a procedural-state problem, not mainly an IRAC problem.

### CJPE 1961_417 (gold: rejected)

The case concerns whether an illegitimate son could inherit an additional share after a widow's death. The paper reports two expert errors. One expert relied too heavily on a cited precedent that the Supreme Court did not consider relevant. Another correctly understood the legal proposition but attributed it to the wrong party: the court recognized the entitlement, while the appellant was contesting that entitlement. The expert therefore inverted the outcome. Both BF16 CJPE conditions reproduced this party-stance error: they correctly stated the inheritance rule but predicted acceptance. The paired facts-only conditions labeled the case correctly, showing that additional full-document reasoning can introduce a stance inversion rather than always help.

Motivated trajectory: identify parties and requested relief -> decompose the inheritance issue -> filter precedents for actual relevance -> apply the rule -> check whether the accepted legal proposition supports or defeats the appellant -> disposition. This case needs party-stance tracking and precedent relevance, neither of which is central in case 1962_47.

### CJPE 1954_13 (gold: rejected)

Four of five legal experts predicted acceptance. The paper explains that the case contained multiple issues and associated prayers; the court accepted some arguments and rejected others, creating ambiguity about how to aggregate the final outcome. Both BF16 conditions nevertheless predicted the gold rejected label in both the CJPE and facts-only views.

Motivated trajectory: decompose issues and prayers -> determine the court's ruling on each -> identify which rulings control the benchmark label -> aggregate issue-level outcomes -> final disposition. This same aggregation module is likely important for AnnoCaseLaw's mixed class.

Together, these cases support a modular trajectory library. The modules are reusable, but a planner must select and order them based on the case. Forcing all cases through the same IRAC sequence cannot express this variation reliably.

## 6. Full-BF16 rationale error patterns

The audit below now uses the complete Delta BF16 scored ledger: all 6,840 direct and structured generations. The paired outcomes confirm that the prompts mostly trade cases rather than improve them consistently. AnnoCaseLaw has 39 direct-only wins and 41 structured-only wins; Realistic LJP has 44 and 34; CJPE has 66 and 59.

- Procedural-layer copying: in AnnoCaseLaw 0029, both prompts reproduced the intermediate Court of Appeals' policy rationale and treated it as the present court's conclusion. A trajectory should first record the current tribunal, challenged holding, and standard of review.
- Current-proceeding confusion: in facts-only case 1962_47, direct predicted acceptance because the text said an earlier civil appeal had been allowed. Structured correctly recognized that the present matter was a review petition and predicted rejection. Both CJPE versions were correct because the longer proceedings made the procedural state clearer.
- Intermediate grant versus final outcome: in facts-only case 1961_170, structured treated the grant of special leave and framing of a constitutional question as evidence that the appeal was accepted, while direct correctly predicted the final rejection. A procedural-event taxonomy is needed so “leave granted,” “appeal heard,” and “relief allowed” are not collapsed.
- Party-stance inversion: in CJPE 1961_417, both prompts correctly concluded that the illegitimate son was entitled to inherit but failed to notice that the appellant opposed that entitlement, so both predicted acceptance instead of rejection. Two legal experts made the same error. The missing operation is not more doctrine; it is mapping each accepted proposition back to the party seeking relief.
- Rationale–label polarity mismatch: in CJPE 1999_721, structured correctly reasoned that the challenged tax proviso lacked an ascertainable rate and rejected the High Court's interpretation, but its final label was “rejected.” Direct mapped the same reasoning to the correct accepted label. A final entailment check should ask whether the stated holdings support or defeat the appellant.
- Doctrine-specific misapplication under fixed structure: in AnnoCaseLaw 0001, both prompts misapplied exclusive control and the evidentiary role of res ipsa loquitur. Merely enumerating IRAC elements did not prevent the wrong evidence-to-element mapping.
- Over-deference to the observed lower-court result: in AnnoCaseLaw 0510, direct correctly found genuine disputes over negligence and reversed summary judgment. Structured discounted those disputes and affirmed largely because the trial court had already granted summary judgment.
- Failure to represent mixed outcomes: neither BF16 condition predicted “mixed” in any of the 394 AnnoCaseLaw cases, so all 57 gold-mixed cases were missed. Issue/prayer decomposition followed by an explicit aggregation rule is necessary for this class.

The full-run evidence narrows the claim: a generic structured prompt is not a reliable remedy. Useful structure must be selected by case—procedural-state tracking for one case, party stance for another, doctrine-specific elements for a third, and issue aggregation for mixed outcomes—then followed by a label-consistency check.

## 7. Implications for LegalFlux

The case evidence supports a case-adaptive, step-wise planner with reusable modules rather than one fixed legal template. A practical initial module set is:

- proceeding identity and procedural posture;
- party, claim, and requested-relief tracking;
- issue and prayer decomposition;
- material-fact filtering;
- doctrine or rule identification;
- doctrine-specific element checking;
- precedent relevance and stance checking;
- judge-versus-jury or standard-of-review allocation;
- issue-level outcome aggregation, including mixed outcomes;
- final rationale–label consistency verification.

Not every case should invoke every module. The planner's task is to select the smallest sufficient sequence, order it appropriately, and reuse it on legally similar X_sim cases. The current DPO design directly tests this claim: a trajectory is rewarded only if its fixed sequence transfers from the anchor to two similar cases, which discourages a plan that merely overfits one document.

## 8. Next steps

- Complete the DPO candidate and X_sim evaluation ledgers; report pair yield, reward margins, plan lengths, and the frequency of all-tie groups before starting training.
- Build an aligned evaluation set from the locally available 56-case ILDCexpert ranked explanations and the paired CJPE/facts views. Score decisive-evidence coverage, party/proceeding consistency, and unsupported steps without treating importance ranks as a unique gold trajectory.
- Convert AnnoCaseLaw application spans and CJPE ranked sentences into a small human-audited trajectory-evaluation set. Use them to score step coverage, ordering plausibility, and unsupported steps—not to claim a unique gold trajectory.
- Use the 1,509 exactly paired CJPE/facts cases for controlled input ablations, including rhetorical-role and late-document ablations, to distinguish genuine legal reasoning gains from outcome-adjacent cues.
- Add targeted ablations for the most strongly motivated modules: procedural-state tracking, party/stance tracking, issue aggregation, and final consistency checking.

## Evidence sources

- Internal full benchmark run: delta-bf16-three-dataset-full-gpub017-v1.
- AnnoCaseLaw paper and dataset.
- [ILDC for CJPE paper (ACL 2021)](https://aclanthology.org/2021.acl-long.313/).
- [Realistic LJP paper (NLLP 2024)](https://aclanthology.org/2024.nllp-1.6/).
- LegalFlux DPO implementation and cluster workflow on main.
