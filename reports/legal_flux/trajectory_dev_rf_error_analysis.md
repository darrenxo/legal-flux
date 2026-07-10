# LegalFlux RF-Style Trajectory-Dev Error Analysis

This report uses the latest run hashes in `runs/legal_flux/trajectory_dev/run_plan.json`.
Model-facing inputs in this run used plaintiff claim/task, parties, facts, and supplied authority context.
Scored rows: 767. Error rows excluded from metric tables: 1.

## Condition Summary
| condition | n | answer acc | binary valid | issue coverage | avg calls | avg sec | avg prompt tok | avg output tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | 256 | 68.0% | 100.0% | 0.0% | 1.00 | 1.49 | 693 | 99 |
| structured | 256 | 59.0% | 100.0% | 92.7% | 1.00 | 2.88 | 874 | 230 |
| flux_rf_style | 255 | 67.1% | 100.0% | 0.0% | 8.65 | 37.54 | 17978 | 2972 |

## Direct vs RF Outcome Buckets
| bucket | count | share |
| --- | --- | --- |
| both_correct | 127 | 49.8% |
| direct_only | 46 | 18.0% |
| adaptive_only | 44 | 17.3% |
| both_wrong | 38 | 14.9% |

## Pairwise Overlap
| pair | n | both correct | left only | right only | both wrong | prediction disagreements |
| --- | --- | --- | --- | --- | --- | --- |
| direct vs flux_rf_style | 255 | 127 | 46 | 44 | 38 | 90 |
| direct vs structured | 256 | 127 | 47 | 24 | 58 | 71 |
| structured vs flux_rf_style | 255 | 100 | 51 | 71 | 33 | 122 |

## Prediction Distribution
| condition | support | reject | mixed | unresolved | none | non-binary share |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 108 | 148 | 0 | 0 | 0 | 0.0% |
| structured | 59 | 197 | 0 | 0 | 0 | 0.0% |
| flux_rf_style | 175 | 80 | 0 | 0 | 0 | 0.0% |

| condition | gold | n | pred support | pred reject | gold-specific acc |
| --- | --- | --- | --- | --- | --- |
| direct | support | 128 | 77 | 51 | 60.2% |
| direct | reject | 128 | 31 | 97 | 75.8% |
| structured | support | 128 | 41 | 87 | 32.0% |
| structured | reject | 128 | 18 | 110 | 85.9% |
| flux_rf_style | support | 127 | 109 | 18 | 85.8% |
| flux_rf_style | reject | 128 | 66 | 62 | 48.4% |

## RF Failure Subtypes
| adaptive wrong subtype | count | share of adaptive wrong |
| --- | --- | --- |
| opposite_binary_prediction | 84 | 100.0% |
| false_support | 66 | 78.6% |
| structured_was_correct | 51 | 60.7% |
| direct_was_correct | 46 | 54.8% |
| false_reject | 18 | 21.4% |

## Trajectory Correlates
| bucket | n | avg trajectory len | avg reviews | avg calls | avg repair count | binary valid |
| --- | --- | --- | --- | --- | --- | --- |
| rf_correct | 171 | 3.82 | 3.82 | 8.64 | 2.51 | 100.0% |
| rf_wrong | 84 | 3.83 | 3.83 | 8.67 | 2.69 | 100.0% |
| rf_false_support | 66 | 3.80 | 3.80 | 8.61 | 2.61 | 100.0% |
| rf_false_reject | 18 | 3.94 | 3.94 | 8.89 | 3.00 | 100.0% |

## Lawsuit Type Deltas
| lawsuit_type | n | direct acc | structured acc | rf acc | rf-direct |
| --- | --- | --- | --- | --- | --- |
| civil action | 6 | 50.0% | 16.7% | 100.0% | +50.0 pp |
| judicial review application | 6 | 50.0% | 33.3% | 100.0% | +50.0 pp |
| (blank) | 23 | 56.5% | 69.6% | 60.9% | +4.3 pp |
| Judicial review application | 7 | 85.7% | 85.7% | 85.7% | +0.0 pp |
| Personal Injuries Action | 5 | 100.0% | 80.0% | 80.0% | -20.0 pp |
| Magistracy Appeal | 5 | 100.0% | 80.0% | 0.0% | -100.0 pp |

## Retrieval And Template Diagnostics
**Retrieval modes**
| mode | count |
| --- | --- |
| embedding_ambiguous_exact | 731 |
| exact_unique | 240 |
| embedding_full_pool | 4 |

**Similarity summary:** count=975, mean=0.788, min=0.512, max=1.000
**Cases with repeated template IDs in one trajectory:** 0/255

**Trajectory length distribution**
| length | count |
| --- | --- |
| 1 | 1 |
| 3 | 42 |
| 4 | 212 |

**Review count distribution**
| reviews | count |
| --- | --- |
| 1 | 1 |
| 3 | 42 |
| 4 | 212 |

**Lowest RF template-associated accuracies, min 8 uses**
| template | uses | case acc when used |
| --- | --- | --- |
| LF007 - Reconcile Overlapping Statutory and Procedural Regimes | 33 | 54.5% |
| LF014 - Assess Credibility and Discharge of the Civil Burden | 31 | 54.8% |
| LF008 - Resolve Competing Legal Characterizations | 50 | 62.0% |
| LF005 - Rule-to-Element Extraction and Mapping | 70 | 62.9% |
| LF002 - Competing Narrative and Evidence Reliability Resolution | 41 | 63.4% |
| LF004 - Multi-Source Rule Synthesis | 69 | 63.8% |
| LF003 - Identify and Complete the Governing Doctrine | 216 | 66.2% |
| LF001 - Issue-Directed Long-Fact Filtering | 240 | 67.5% |
| LF006 - Temporal Operation of Legal Rules | 18 | 72.2% |
| LF012 - Element-Wise Burden and Evidence Matrix | 102 | 72.5% |
| LF010 - Select the Closest Material Analogy | 40 | 72.5% |
| LF071 - Statutory Classification of Property or Use | 10 | 80.0% |
| LF009 - Extract and Apply a Precedential Principle | 37 | 81.1% |

**Highest RF template-associated accuracies, min 8 uses**
| template | uses | case acc when used |
| --- | --- | --- |
| LF009 - Extract and Apply a Precedential Principle | 37 | 81.1% |
| LF071 - Statutory Classification of Property or Use | 10 | 80.0% |
| LF010 - Select the Closest Material Analogy | 40 | 72.5% |
| LF012 - Element-Wise Burden and Evidence Matrix | 102 | 72.5% |
| LF006 - Temporal Operation of Legal Rules | 18 | 72.2% |
| LF001 - Issue-Directed Long-Fact Filtering | 240 | 67.5% |
| LF003 - Identify and Complete the Governing Doctrine | 216 | 66.2% |
| LF004 - Multi-Source Rule Synthesis | 69 | 63.8% |
| LF002 - Competing Narrative and Evidence Reliability Resolution | 41 | 63.4% |
| LF005 - Rule-to-Element Extraction and Mapping | 70 | 62.9% |
| LF008 - Resolve Competing Legal Characterizations | 50 | 62.0% |
| LF014 - Assess Credibility and Discharge of the Civil Burden | 31 | 54.8% |
| LF007 - Reconcile Overlapping Statutory and Procedural Regimes | 33 | 54.5% |

**Most common planned step names**
| step name | count |
| --- | --- |
| Issue-Directed Long-Fact Filtering | 120 |
| Identify and Complete the Governing Doctrine | 119 |
| Element-Wise Burden and Evidence Matrix | 75 |
| Issue-Directed Long-Fact Filtering and Chronology Construction | 55 |
| Select the Closest Material Analogy | 40 |
| Multi-Source Rule Synthesis | 36 |
| Rule-to-Element Extraction and Mapping | 33 |
| Resolve Competing Legal Characterizations | 23 |
| Competing Narrative and Evidence Reliability Resolution | 14 |
| Reconcile Overlapping Statutory and Procedural Regimes | 13 |
| Issue-Directed Long-Fact Filtering and Chronology Reconstruction | 4 |
| Element-Wise Burden and Evidence Matrix Construction | 4 |
| Multi-Source Rule Synthesis and Element Mapping | 4 |
| Multi-Source Rule Synthesis and Precedent Application | 4 |
| Temporal Operation of Legal Rules | 4 |

**Most common planned tags**
| tag | count |
| --- | --- |
| fact_filtering | 234 |
| doctrine_identification | 218 |
| rule_recall | 174 |
| material_facts | 158 |
| element_mapping | 141 |
| legal_classification | 136 |
| precedent | 127 |
| burden | 123 |
| statutory_interpretation | 122 |
| credibility | 102 |
| rule_application | 90 |
| chronology | 80 |
| evidence_organization | 69 |
| rule_extraction | 68 |
| legal_characterization | 67 |
| legal_test | 58 |
| analogy | 58 |
| substance_over_form | 57 |
| documentary_evidence | 55 |
| procedural_route | 39 |

## Examples: Direct Correct, RF Wrong
| case | gold | direct | rf | structured | lawsuit_type | traj len | reviews | planned steps | templates | rf rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legalhk-10695 | reject | reject | support | support | action for damages | 4 | 4 | Issue-Directed Long-Fact Filtering: Characterize Breach Severity and Right to Reject;Competing Narrative and Evidence... | Issue-Directed Long-Fact Filtering;Competing Narrative and Evidence Reliability Resolution;Multi-Source Rule Synthesis;Rule-to-Element Extraction and Mapping | The evidence supports a finding in favor of the Plaintiff's right to recover damages. The supply of 1.4D fiber instead of contracted 1.5D fiber constitutes a fundamental breach under Section 15(1) (F7, F9), justifying... |
| legalhk-11327 | reject | reject | support | reject | Appeal against summary judgment | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mappin... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mapping;Reconcile Overlapping Statutory and Pr... | The Plaintiff's claim for rescission is supported because the Defendant failed to sell the property free from defects (unauthorized building works), which constitutes a repudiatory breach under Clause 19 as these are ... |
| legalhk-11956 | reject | reject | support | reject | Judicial review application | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Element-W... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Element-Wise Burden and Evidence Matrix | The application for judicial review should be supported (or the challenge to disclosure upheld) because the Defendant has satisfied its burden under Order 53 rule 1A. The evidence in F6 and F7 demonstrates that disclo... |
| legalhk-12205 | reject | reject | support | reject | application to set aside default judgment | 4 | 4 | Issue-Directed Long-Fact Filtering and Chronology Construction;Identify and Complete the Governing Doctrine on Confid... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Resolve Competing Legal Characterizations | The executed artifacts demonstrate that the 40th defendant's application should be supported. The facts (F1-F10) confirm the existence of confidential information and a duty of confidence, but also show that the docum... |
| legalhk-12729 | reject | reject | support | reject | (blank) | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mappin... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mapping;Resolve Competing Legal Characterizations | The claim is supported because the facts demonstrate a valid 'account stated' under Hong Kong law. F5 establishes a calculated net sum due after separating accounts in 1996, creating an existing liability. F6 and F8 s... |
| legalhk-12988 | reject | reject | support | reject | pre-trial review | 4 | 4 | Issue-Directed Long-Fact Filtering and Chronology Construction;Identify and Complete the Governing Doctrine for Contr... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mapping;Resolve Competing Legal Characterizations | The counterclaim is supported. It arises from the same factual matrix as the claim (F6), was amended appropriately following the service of witness statements in October 2002 (F3) to reflect crystallized evidence, and... |
| legalhk-13337 | reject | reject | support | reject | Summary Judgment Application, Application for Continuation of Injunction, Application for Security for Costs, and Application for Leave to Amend Defence | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Element-W... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Element-Wise Burden and Evidence Matrix | The claim for restitution is supported. Facts F3, F6, and F7 establish that the payment was made under a fundamental mistake induced by fraud/scam emails, negating any consent or legal basis. Fact F4 confirms the defe... |
| legalhk-13695 | reject | reject | support | reject | civil | 4 | 4 | Fact Filtering and Issue Framing: Determine if funds were held on trust or as a loan given the specific purpose of fi... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Assess Credibility and Discharge of the Civil Burden;Rule-to-Element Extracti... | The Plaintiff is entitled to recovery of the US$95,000 because the primary purpose for which funds were transferred (production of 'Kwailo') failed entirely. Under the principles in Fibrosa and Quistclose, retention o... |
| legalhk-13830 | support | support | reject | reject | Magistracy Appeal | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Element-W... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Element-Wise Burden and Evidence Matrix | The appeal is supported because the evidence does not conclusively establish that the Appellant suffers from an organic 'mental disorder' under s.16(4) of the Mental Health Ordinance sufficient to justify a hospital o... |
| legalhk-13888 | reject | reject | support | reject | Appeal under section 6(9) of the Legal Practitioners Ordinance | 4 | 4 | Issue-Directed Long-Fact Filtering and Chronology Construction;Identify and Complete the Governing Doctrine on Specia... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matrix;Resolve Competing Legal Characterizat... | The decision is supported because F3 confirms appellants' conduct exposed clients to systemic failings, and S02/S04 establish that this exposure justifies conditions under s.6 LPO regardless of non-managerial status (... |

## Examples: RF Correct, Direct Wrong
| case | gold | direct | rf | structured | lawsuit_type | traj len | reviews | planned steps | templates | rf rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legalhk-10621 | support | reject | support | reject | judicial review application | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matr... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matrix;Resolve Competing Legal Characterizat... | The Plaintiff should be supported because the Director of Immigration erred in law by applying policy rigidity over substantive humanitarian necessity. The facts establish that the sister suffers from severe mental in... |
| legalhk-11275 | support | reject | support | reject | Equitable Lien Claim | 4 | 4 | Issue-Directed Long-Fact Filtering: Distinguish Void Ab Initio vs. Rescission Claims Against Subsequent Purchasers;Id... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Multi-Source Rule Synthesis;Assess Credibility and Discharge of the Civil Burden | The trajectory concludes that the assignment from Plaintiff to Defendant 1 was void ab initio due to illegality (F6, F7) involving a scheme with an indigenous villager and tainted consideration. Under property law pri... |
| legalhk-11353 | support | reject | support | reject | judicial review application | 4 | 4 | Issue-Directed Long-Fact Filtering: Isolate Zoning Classification Dispute from Historical Context Claims;Identify and... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Extract and Apply a Precedential Principle;Resolve Competing Legal Characteri... | The CPLD's decisions should be quashed and mandamus issued because the Committee acted unreasonably by relying on historical context (F6) rather than current development realities. The Applicants demonstrated that the... |
| legalhk-11938 | support | reject | support | reject | Application for determination of a question of law and summary judgment | 3 | 3 | Issue-Directed Long-Fact Filtering and Chronology Construction;Identify and Complete the Governing Doctrine on Revoca... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Competing Narrative and Evidence Reliability Resolution | Supporting the Plaintiff's claim: The analysis confirms that under s.13(1) of the Wills Ordinance, a will is revoked by any subsequent marriage 'in effect' at death (F4), not merely one occurring later in time. Step S... |
| legalhk-12095 | reject | support | reject | reject | (blank) | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matr... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matrix;Select the Closest Material Analogy | The claim for payment must be rejected. The cheques were issued blank to an employee (KONG Sau-sang) with a specific limited purpose: filling them only after tour costs were incurred and services rendered (F1, F2). KO... |
| legalhk-1235 | support | reject | support | reject | civil action | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mappin... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mapping;Reconcile Overlapping Statutory and Pr... | Support is granted because the executed analysis establishes that deposits paid under a terminated franchise agreement are subject to restitution principles where consideration fails or becomes unjust. The Defendant's... |
| legalhk-12423 | support | reject | support | support | (blank) | 4 | 4 | Fact Filtering and Issue Framing: Vacant Possession Definition;Chronology and Evidence Organization: Timeline of Rein... | Issue-Directed Long-Fact Filtering;Competing Narrative and Evidence Reliability Resolution;Identify and Complete the Governing Doctrine;Rule-to-Element Extra... | The claim is supported. The defendant failed to deliver vacant possession on the tenancy expiry date (14th February 2003) because the premises were not reinstated until August/September 2003, as established by facts F... |
| legalhk-1248 | support | reject | support | reject | judicial review application | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matr... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Element-Wise Burden and Evidence Matrix;Select the Closest Material Analogy | Support is granted because the alternative accommodation fails the 'necessity' threshold. The proposed living space of approximately 50 sq. ft. without windows (F3) renders it uninhabitable, distinguishing this case f... |
| legalhk-12483 | support | reject | support | reject | Mareva application and discovery summons | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Competing Narrative and Evidence Reli... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Competing Narrative and Evidence Reliability Resolution;Rule-to-Element Extra... | The claim is supported. The executed artifacts confirm an agency agreement (F1) where the defendant failed to remit proceeds after placing them in inaccessible time deposits (F4, F5). Evidence reliability analysis (S3... |
| legalhk-12760 | reject | support | reject | reject | Application for relief from forfeiture | 4 | 4 | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mappin... | Issue-Directed Long-Fact Filtering;Identify and Complete the Governing Doctrine;Rule-to-Element Extraction and Mapping;Select the Closest Material Analogy | The Defendant's claim for possession and relief from forfeiture must be rejected. The parties reached a consent agreement on 20 March 1984 (F6) containing an undertaking to vacate the premises and pay arrears, explici... |

## Artifacts
- Case-level delta CSV: `reports\legal_flux\trajectory_dev_rf_case_deltas.csv`
- Scored rows: `runs\legal_flux\trajectory_dev\scored.jsonl`
- Aggregate metrics: `runs\legal_flux\trajectory_dev\aggregate.csv`
