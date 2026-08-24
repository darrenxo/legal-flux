# LegalHK trajectory-dev complexity analysis

Paired cases: 2,755. Fact-count and gold-text-length bins are approximately equal-frequency quartiles. Issue-count bins are 1, 2, 3, and 4+. Cases missing a measure are excluded only from that measure.

## Number of facts

| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1: 2–9 facts | 1,101 | 74.39% | 74.59% | 66.49% | 66.43% | -7.90% |
| Q2: 10 facts | 402 | 71.89% | 72.09% | 65.92% | 65.67% | -5.97% |
| Q3: 11–12 facts | 635 | 77.64% | 77.85% | 71.34% | 71.53% | -6.30% |
| Q4: 13–50 facts | 617 | 75.69% | 76.03% | 68.40% | 68.81% | -7.29% |

## Number of issues

| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 375 | 70.93% | 70.96% | 66.67% | 66.06% | -4.27% |
| 2 | 1,156 | 75.87% | 76.18% | 69.98% | 70.35% | -5.88% |
| 3 | 771 | 75.88% | 76.08% | 67.57% | 67.72% | -8.30% |
| 4+ | 388 | 78.09% | 78.20% | 67.78% | 67.77% | -10.31% |

## Gold court-reasoning + judgment length

| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1: 12–136 words | 690 | 74.06% | 74.68% | 66.67% | 68.11% | -7.39% |
| Q2: 137–166 words | 696 | 75.14% | 75.83% | 66.09% | 66.94% | -9.05% |
| Q3: 167–202 words | 690 | 77.25% | 77.32% | 69.13% | 68.98% | -8.12% |
| Q4: 203–548 words | 679 | 73.78% | 73.78% | 69.96% | 68.80% | -3.83% |
