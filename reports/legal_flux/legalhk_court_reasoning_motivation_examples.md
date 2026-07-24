# LegalHK Court-Reasoning Examples for LegalFlux Motivation

## Dataset Signals

- Rows inspected: 18,374
- Non-empty court reasoning rows: 18,373
- Rows with supplied related laws: 14,324
- Rows with supplied relevant cases: 13,468
- Rows with estimated multiple issues: 15,339

## Regex Pattern Counts

- `precedent_analogy`: 9,072
- `remedy_or_discretion`: 8,192
- `statutory_or_rule_application`: 4,963
- `multi_issue_composition`: 4,549
- `evidence_credibility_burden`: 3,542
- `procedural_gateway`: 3,250
- `injunction_discretion`: 729

## Most Frequent Pattern Combinations

- `none`: 2,122
- `precedent_analogy`: 1,842
- `remedy_or_discretion`: 1,499
- `precedent_analogy|remedy_or_discretion`: 1,208
- `statutory_or_rule_application|precedent_analogy`: 718
- `evidence_credibility_burden`: 683
- `statutory_or_rule_application|precedent_analogy|remedy_or_discretion`: 609
- `multi_issue_composition`: 549
- `precedent_analogy|multi_issue_composition`: 520
- `statutory_or_rule_application|remedy_or_discretion`: 505
- `statutory_or_rule_application`: 498
- `evidence_credibility_burden|precedent_analogy`: 457

## Candidate Introduction Examples

### legalhk-1864 — Summary Judgment Application

- Gold label: `support`
- Estimated issues: 2; related laws: 3; relevant cases: 0
- Matched high-level patterns: `procedural_gateway; precedent_analogy; remedy_or_discretion`
- Claim: HK$229,390.56 being the deficit of price for resale of the goods as a result of non-acceptance of goods by the Defendant
- Issues: Whether the Defendant has a fair or reasonable probability of showing a real or bona fide defence Whether the Plaintiff has proved its case beyond reasonable doubt
- Court-reasoning excerpt: “The Court applied the principles of summary judgment as stated in Order 14 of the Rules of the District Court, Cap.336 The Court considered the Defendant's defence to be inconsistent and not reasonably capable of...”

### legalhk-9741 — application to stay proceedings

- Gold label: `reject`
- Estimated issues: 2; related laws: 2; relevant cases: 3
- Matched high-level patterns: `procedural_gateway; precedent_analogy`
- Claim: fraudulent misappropriation of HK$130,000,000 odd
- Issues: Whether the defendant’s application to stay the civil proceedings should be granted. Whether the defendant’s right to remain silent would be compromised if the civil proceedings are not stayed.
- Court-reasoning excerpt: “The court considered the defendant's application to stay the civil proceedings in light of the concurrent criminal proceedings. The court applied the principles set out in Jefferson v Bhetcha [1979] 1 WLR 898, which ...”

### legalhk-11593 — judicial review application

- Gold label: `support`
- Estimated issues: 1; related laws: 2; relevant cases: 2
- Matched high-level patterns: `evidence_credibility_burden; statutory_or_rule_application; precedent_analogy`
- Claim: Production and inspection of documents pursuant to section 121 or section 152FA of the Companies Ordinance, Cap 32
- Issues: Whether the plaintiffs are entitled to production and inspection of the documents sought pursuant to section 121 or section 152FA of the Companies Ordinance, Cap 32.
- Court-reasoning excerpt: “... inspection. A shareholder is entitled to inspect documents if he subjectively believes that he is inspecting for a proper purpose and the court is objectively satisfied that the inspection sought is for a proper purpose. The plaintiffs have made out a very strong case for production and inspection of the documents sought. Thei...”

### legalhk-6761 — Companies (Winding Up and Miscellaneous Provisions) proceedings

- Gold label: `support`
- Estimated issues: 2; related laws: 1; relevant cases: 0
- Matched high-level patterns: `procedural_gateway; evidence_credibility_burden`
- Claim: To stay further winding-up proceedings permanently in exchange for paying the liquidators' fees and expenses
- Issues: Whether a permanent stay of the winding-up proceedings should be granted Whether fees are payable upon the value of the property on an ad valorem basis
- Court-reasoning excerpt: “The court considered that the applicants have provided genuine commercial reasons for a stay of the winding-up proceedings, namely to allow the company to dispose of its property in Shenzhen. The court was satisfied that there appear to be no further o...”

### legalhk-13093 — judicial review

- Gold label: `reject`
- Estimated issues: 4; related laws: 6; relevant cases: 3
- Matched high-level patterns: `statutory_or_rule_application; precedent_analogy; remedy_or_discretion`
- Claim: The applicants claim that they are beneficiaries of the Court of Final Appeal judgments in Ng Ka Ling and Chan Kam Nga and that they have legitimate expectations that the Government will implement those judgments in their favour.
- Issues: Whether the applicants are beneficiaries of the Court of Final Appeal judgments in Ng Ka Ling and Chan Kam Nga Whether the applicants have legitimate expectations that the Government will implement those judgments in their favour Whether the Director of Immigration has misinterpreted the Concession and treated the applicants unfairly Whether the removal orders were made in disregard of the legitimate expectations that the applicants had acquired under the Court of Final Appeal judgments
- Court-reasoning excerpt: “...urt rejected the applicants' argument that the Director of Immigration had misinterpreted the Concession and treated them unfairly, holding that the Director's interpretation of the Concession was reasonable and in line with the Government's policy. The court also rejected the applicants' argument that the removal orders were m...”

### legalhk-6355 — (blank lawsuit_type)

- Gold label: `support`
- Estimated issues: 2; related laws: 0; relevant cases: 0
- Matched high-level patterns: `evidence_credibility_burden`
- Claim: The plaintiff claims the equivalent of a sum of £1,000 which was admittedly paid by him to the defendant.
- Issues: The main issue in dispute is the purpose for which the £1,000 was sent to the defendant. The court must determine whether the defendant was entitled to regard the balance of the £1,000 as part of the partnership funds.
- Court-reasoning excerpt: “.... The court must determine whether the defendant was entitled to regard the balance of the £1,000 as part of the partnership funds. The court must consider the credibility of the parties and their versions of events. The court must consider the documentary evidence, including the correspondence between the parties.”

### legalhk-14381 — Action No 35 of 2015

- Gold label: `support`
- Estimated issues: 3; related laws: 0; relevant cases: 80
- Matched high-level patterns: `injunction_discretion; remedy_or_discretion`
- Claim: claimed for the loan of RMB 4.5 million with interest at 4.5% per month
- Issues: Whether Sung Ngai Yeung had repaid the loan to the plaintiff. Whether the Alleged Settlement Agreement existed and discharged Sung Ngai Yeung's liability to the plaintiff. Whether there was a risk of dissipation of Sung Ngai Yeung's assets if a Mareva injunction was not granted.
- Court-reasoning excerpt: “...tiff's claim for interlocutory relief under Order 29 of the Rules of the High Court and the lack of representation by Sung Ngai Yeung in determining the Mareva injunction application.”

### legalhk-13583 — appeal against the order of the Master, who refused to enter summary judgment on the plaintiff's claim and granted unconditional leave to the defendant to defend the action

- Gold label: `reject`
- Estimated issues: 3; related laws: 122; relevant cases: 3
- Matched high-level patterns: `procedural_gateway; evidence_credibility_burden; precedent_analogy; multi_issue_composition`
- Claim: claims that the defendant, Chan Koon Chow, is in breach of an agreement dated 28 May 2015, which included the grant of the Exclusive Wholesale Distribution Right of Gucci merchandise in Mainland China, Thailand, Singapore, and Taiwan
- Issues: Whether the defendant has raised a credible defence to the plaintiff's claim Whether the Incorp Agreement is effective in granting the Exclusive Right for the distribution of Gucci products in the Territory Whether the defendant has discharged his obligations under the Agreement
- Court-reasoning excerpt: “...e defence against the evidence disclosed, and a mini trial on factual disputes will not be conducted The Incorp Agreement is not clear on its face, and it is a question for trial whether Incorp has rights in the Gucci trademark or brand name to grant the Exclusive Right There is no evidence that Incorp is, or is not, a company...”
