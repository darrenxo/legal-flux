# LegalHK trajectory-dev complexity analysis

Paired cases: 2,755. Fact-count bins are 2–5, 6–10, 11–15, and 16+. Issue-count bins are 1, 2, 3, and 4+. Court-reasoning-only length uses approximately equal-frequency quartiles. Cases missing a measure are excluded only from that measure.

## Number of facts

| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2–5 | 68 | 75.00% | 75.11% | 69.12% | 68.86% | -5.88% |
| 6–10 | 1,435 | 73.66% | 73.86% | 66.20% | 66.10% | -7.46% |
| 11–15 | 1,039 | 77.00% | 77.25% | 70.36% | 70.59% | -6.64% |
| 16+ | 213 | 75.12% | 75.51% | 67.61% | 68.21% | -7.51% |

## Number of issues

| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 375 | 70.93% | 70.96% | 66.67% | 66.06% | -4.27% |
| 2 | 1,156 | 75.87% | 76.18% | 69.98% | 70.35% | -5.88% |
| 3 | 771 | 75.88% | 76.08% | 67.57% | 67.72% | -8.30% |
| 4+ | 388 | 78.09% | 78.20% | 67.78% | 67.77% | -10.31% |

## Gold court-reasoning length

| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1: 9–97 words | 689 | 73.73% | 73.92% | 68.07% | 67.81% | -5.66% |
| Q2: 98–119 words | 696 | 74.43% | 74.72% | 66.95% | 66.95% | -7.47% |
| Q3: 120–146 words | 690 | 74.06% | 74.22% | 69.28% | 69.52% | -4.78% |
| Q4: 147–358 words | 680 | 78.09% | 78.36% | 67.50% | 67.86% | -10.59% |
