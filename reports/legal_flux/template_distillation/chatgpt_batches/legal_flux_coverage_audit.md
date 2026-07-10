# LegalFlux Final Template-Pool Coverage Audit

## Overall judgment

**Adequate with targeted revision.** The final pool contains 116 schema-valid templates with unique IDs and broad coverage of the dominant reasoning demands and legal families. The strongest remaining gaps concern reasoning control rather than major doctrinal areas: focused single-issue resolution, explicit dual-issue resolution, cross-domain interfaces, and generic affirmative-defense testing. Probate also needs more specific coverage of testamentary instruments and estate distribution.

## Structural checks

- 116 JSONL records parsed successfully.
- 116 unique `template_id` values.
- 0 schema-validation errors against `legal_flux_template.schema.json`.
- The batch manifest contains 778 batch slots but only 703 unique case IDs, covering 50.2% of the 1,400 source cases; 75 slots repeat cases used elsewhere. Coverage conclusions should therefore be treated as coverage of the observed/distilled sample, not proof of full corpus saturation.

## 1. Covered categories

### General reasoning demands
- Multi-issue decomposition and decision sequencing: LF001, LF022.
- Long-fact and chronology filtering: LF002-LF004.
- Supplied-rule extraction, doctrine identification, and source synthesis: LF005-LF011.
- Precedent extraction, analogy selection, and distinguishing: LF012-LF014.
- Evidence, burden, credibility, and missing-evidence analysis: LF015-LF018.
- Claim/defense/counterclaim framing, legal status, and entitlement-before-quantum: LF019-LF021.

### Procedure and remedies
- Jurisdiction, standing, service, default relief, summary disposition, strike-out, amendments, disclosure, appeal, judicial review, interim relief, stays, declarations, and remedy calibration: LF023-LF043.

### Domain families
- Contract performance, payment, debt, damages, arbitration, and coordinated remedies: LF044-LF071.
- Property, possession, valuation, trusts, family status, and monetary provision: LF072-LF085.
- Tort, negligence, accident reconstruction, causation, apportionment, damages, employment compensation, and defamation: LF086-LF105.
- Company and insolvency: LF106-LF116.

These areas align well with the largest observed source families and the highest-frequency demands: contract performance, tort/negligence, property/possession, procedural thresholds, precedent/analogy, evidence/burden, supplied-rule extraction, multi-issue composition, and remedy discretion.

## 2. Under-covered categories

1. **Dual-issue resolution.** It appears 498 times among all reasoning demands, but there is no template dedicated to deciding whether two issues are sequential, independent, alternative, or mutually exclusive.
2. **Focused single-issue resolution.** It appears 203 times, but the pool lacks a template for strict scope control around one narrow issue.
3. **Cross-domain composition.** Recurrent trajectories combine contract-debt, contract-property, contract-tort, property-trust, property-company, tort-employment, and other domain pairs. Current multi-issue templates do not explicitly handle rule priority, displacement, shared facts, or remedy compatibility across domains.
4. **Defense merits and preservation.** Defense/counterargument analysis is frequent, but LF019 mainly classifies claims, defenses, and counterclaims. A general template should test availability, elements, burden, pleading, waiver, and legal effect.
5. **Probate specificity.** LF082-LF085 cover beneficial ownership, administration, family status, and financial relief, but not testamentary validity/construction or the priority-and-distribution sequence of estate administration.
6. **Public/criminal/immigration.** No dedicated coverage is present, but only two source cases fall in this family. The evidence is too sparse to justify a stable template now; flag it for future resampling rather than filling it speculatively.

## 3. Duplicative templates to merge

1. **LF001 + LF022** — both perform dependency-aware multi-issue decomposition and recomposition. Keep one template and preserve LF022's consistency/double-counting checks.
2. **LF002 + LF003** — both create an issue-linked material chronology from a long record. Retain one general template; keep contract examples as an application note.
3. **LF005 + LF008** — both turn supplied legal materials into elements, burdens, and a fact map. Merge into one “operative rule extraction and element mapping” template.
4. **LF049 + LF052** — both map contractual duties, conditions, performance, and breach. Merge, while retaining LF049's implied-cooperation/prevention analysis.
5. **LF059 + LF060** — both reconstruct transactions and compute a net debt balance. Merge and retain explicit set-off/counterclaim treatment.
6. **LF043 + LF071** — LF071 is largely a contract-focused specialization of LF043's multi-remedy compatibility and anti-duplication analysis. Merge into one composite-remedy template with domain-specific examples.
7. **LF106 + LF108** — LF106 already includes petitioner standing as a winding-up gateway; absorb LF108's detailed standing checks into LF106.
8. **LF113 + LF116** — both review corporate restructuring/capital changes through statutory compliance, proper purpose, fairness, and creditor safeguards. Merge into one sanction-and-safeguards template.

These merges would reduce the pool from 116 to approximately 108 templates before adding the six proposed gap-fill templates, yielding a cleaner pool of approximately 114 templates.

## 4. Proposed gap-fill templates

Six additions are proposed in `legal_flux_gap_fill_proposed.jsonl`:

- LF117 Focused Single-Issue Resolution with Scope Control
- LF118 Dual-Issue Dependency and Alternative-Ground Resolution
- LF119 Cross-Domain Claim Interface and Rule-Priority Analysis
- LF120 Affirmative Defense Validity, Burden, and Waiver Check
- LF121 Testamentary Instrument Validity and Construction
- LF122 Estate Administration, Priority, and Distribution Sequence

All six records validate against the supplied schema.
