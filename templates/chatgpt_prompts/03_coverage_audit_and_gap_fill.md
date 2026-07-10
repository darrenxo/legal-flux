# Task: Audit final LegalFlux template-pool coverage

You will receive the final template-pool JSONL, the batch manifest, and the
coverage summary below.

Check whether the final pool covers the main observed reasoning families and
demands. Then return a concise audit report with:

1. Covered categories.
2. Under-covered categories.
3. Duplicative templates that should be merged.
4. Up to 20 additional templates if important gaps remain.

If you propose additional templates, return them as JSONL records matching
`legal_flux_template.schema.json` after the audit report.

Coverage summary:

```json
{
  "template_source_cases": 1400,
  "primary_family_counts": {
    "contract_performance": 764,
    "tort_negligence_damage": 207,
    "property_possession": 173,
    "debt_payment": 91,
    "company_insolvency": 80,
    "trust_probate_family": 43,
    "procedure_appeal": 29,
    "general_civil_reasoning": 11,
    "public_criminal_immigration": 2
  },
  "demand_focus_counts": {
    "procedural_threshold_check": 450,
    "precedent_or_analogy_handling": 380,
    "evidence_and_burden_assessment": 253,
    "defense_or_counterargument_check": 106,
    "multi_issue_composition": 66,
    "remedy_discretion_check": 45,
    "supplied_rule_extraction": 44,
    "long_fact_filtering": 33,
    "rule_recall_or_doctrine_identification": 23
  },
  "all_reasoning_demand_counts": {
    "supplied_rule_extraction": 1057,
    "precedent_or_analogy_handling": 976,
    "multi_issue_composition": 696,
    "dual_issue_resolution": 498,
    "remedy_discretion_check": 493,
    "procedural_threshold_check": 450,
    "long_fact_filtering": 398,
    "evidence_and_burden_assessment": 377,
    "rule_recall_or_doctrine_identification": 343,
    "defense_or_counterargument_check": 238,
    "focused_issue_resolution": 203,
    "issue_spotting_gap": 3
  },
  "trajectory_prefix_counts_top50": {
    "case_profile > issue_decomposition > procedural_threshold > rule_extraction > domain_template:contract_performance": 88,
    "case_profile > issue_confirmation > procedural_threshold > rule_extraction > domain_template:contract_performance": 84,
    "case_profile > issue_decomposition > material_fact_filtering > rule_extraction > domain_template:contract_performance": 63,
    "case_profile > issue_confirmation > rule_extraction > domain_template:contract_performance > domain_template:debt_payment": 55,
    "case_profile > issue_decomposition > rule_extraction > domain_template:contract_performance > domain_template:debt_payment": 52,
    "case_profile > issue_decomposition > material_fact_filtering > procedural_threshold > rule_extraction": 42,
    "case_profile > issue_confirmation > material_fact_filtering > rule_extraction > domain_template:contract_performance": 37,
    "case_profile > issue_confirmation > material_fact_filtering > rule_or_doctrine_identification > domain_template:contract_performance": 37,
    "case_profile > issue_confirmation > material_fact_filtering > procedural_threshold > rule_extraction": 36,
    "case_profile > issue_decomposition > material_fact_filtering > rule_or_doctrine_identification > domain_template:contract_performance": 35,
    "case_profile > issue_decomposition > rule_extraction > domain_template:contract_performance > domain_template:property_possession": 26,
    "case_profile > issue_confirmation > rule_extraction > domain_template:contract_performance > domain_template:property_possession": 25,
    "case_profile > issue_confirmation > procedural_threshold > rule_or_doctrine_identification > domain_template:contract_performance": 24,
    "case_profile > issue_confirmation > rule_or_doctrine_identification > domain_template:contract_performance > domain_template:debt_payment": 24,
    "case_profile > issue_decomposition > rule_or_doctrine_identification > domain_template:contract_performance > domain_template:debt_payment": 22,
    "case_profile > issue_decomposition > procedural_threshold > rule_extraction > domain_template:company_insolvency": 20,
    "case_profile > issue_decomposition > rule_extraction > domain_template:tort_negligence_damage > domain_template:employment_compensation": 19,
    "case_profile > issue_decomposition > procedural_threshold > rule_extraction > domain_template:property_possession": 19,
    "case_profile > issue_confirmation > procedural_threshold > rule_extraction > domain_template:property_possession": 18,
    "case_profile > issue_confirmation > material_fact_filtering > rule_extraction > domain_template:property_possession": 17,
    "case_profile > issue_decomposition > procedural_threshold > rule_extraction > domain_template:tort_negligence_damage": 17,
    "case_profile > issue_confirmation > material_fact_filtering > rule_extraction > domain_template:tort_negligence_damage": 16,
    "case_profile > issue_confirmation > rule_extraction > domain_template:property_possession > domain_template:trust_probate_family": 15,
    "case_profile > issue_decomposition > material_fact_filtering > rule_extraction > domain_template:property_possession": 15,
    "case_profile > issue_confirmation > procedural_threshold > rule_extraction > domain_template:procedure_appeal": 14,
    "case_profile > issue_decomposition > material_fact_filtering > rule_or_doctrine_identification > domain_template:tort_negligence_damage": 14,
    "case_profile > issue_confirmation > material_fact_filtering > rule_extraction > domain_template:debt_payment": 13,
    "case_profile > issue_confirmation > rule_extraction > domain_template:property_possession > domain_template:company_insolvency": 13,
    "case_profile > issue_confirmation > rule_extraction > domain_template:company_insolvency > precedent_or_analogy_check": 13,
    "case_profile > issue_confirmation > procedural_threshold > rule_extraction > domain_template:company_insolvency": 13,
    "case_profile > issue_confirmation > rule_extraction > domain_template:contract_performance > domain_template:tort_negligence_damage": 13,
    "case_profile > issue_decomposition > procedural_threshold > rule_or_doctrine_identification > domain_template:contract_performance": 12,
    "case_profile > issue_decomposition > material_fact_filtering > procedural_threshold > rule_or_doctrine_identification": 11,
    "case_profile > issue_decomposition > material_fact_filtering > rule_extraction > domain_template:tort_negligence_damage": 11,
    "case_profile > issue_confirmation > rule_or_doctrine_identification > domain_template:contract_performance > domain_template:property_possession": 11,
    "case_profile > issue_decomposition > procedural_threshold > rule_extraction > domain_template:procedure_appeal": 11,
    "case_profile > issue_confirmation > rule_extraction > domain_template:trust_probate_family > precedent_or_analogy_check": 10,
    "case_profile > issue_decomposition > rule_extraction > domain_template:tort_negligence_damage > domain_template:company_insolvency": 10,
    "case_profile > issue_decomposition > rule_or_doctrine_identification > domain_template:tort_negligence_damage > domain_template:employment_compensation": 10,
    "case_profile > issue_confirmation > material_fact_filtering > procedural_threshold > rule_or_doctrine_identification": 10,
    "case_profile > issue_confirmation > rule_extraction > domain_template:property_possession > precedent_or_analogy_check": 9,
    "case_profile > issue_decomposition > material_fact_filtering > rule_extraction > domain_template:debt_payment": 9,
    "case_profile > issue_confirmation > rule_extraction > domain_template:tort_negligence_damage > domain_template:employment_compensation": 9,
    "case_profile > issue_confirmation > rule_extraction > domain_template:contract_performance > domain_template:company_insolvency": 9,
    "case_profile > issue_confirmation > rule_extraction > domain_template:debt_payment > domain_template:property_possession": 9,
    "case_profile > issue_decomposition > rule_extraction > domain_template:contract_performance > domain_template:tort_negligence_damage": 9,
    "case_profile > issue_confirmation > material_fact_filtering > rule_or_doctrine_identification > domain_template:tort_negligence_damage": 9,
    "case_profile > issue_decomposition > rule_extraction > domain_template:trust_probate_family > precedent_or_analogy_check": 9,
    "case_profile > issue_decomposition > rule_extraction > domain_template:property_possession > domain_template:tort_negligence_damage": 8,
    "case_profile > issue_confirmation > rule_or_doctrine_identification > domain_template:tort_negligence_damage > precedent_or_analogy_check": 8
  },
  "batch_count": 30,
  "batched_case_ids": 703
}
```
