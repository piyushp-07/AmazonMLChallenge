\# S1 -> S2 Blocking Results



\## Objective



Generate candidate S1 -> S2 entity pairs while maintaining high blocking recall and keeping the candidate set practical for downstream matching.



\## Production Blocking Strategy



The production blocker combines country-aware:



\- Exact normalized business name

\- 4-character normalized name prefix

\- 6-character normalized name prefix

\- Useful business-name token

\- Address token

\- 4-character address prefix

\- 6-character address prefix



Normalization is applied before blocking.



\## Validation Results



\### 5,000 S1 Sample



\- True S1 -> S2 pairs: 8,415

\- Candidates: 2,678,293

\- Recovered: 8,413

\- Missed: 2

\- Blocking recall: 99.98%

\- Candidates/S1: 535.66



\### 20,000 S1 Sample



\- True S1 -> S2 pairs: 33,425

\- Candidates: 42,990,863

\- Recovered: 33,422

\- Missed: 3

\- Blocking recall: 99.9910%

\- Candidates/S1: 2,149.54



\## 2-Character Token Experiment



A 2-character token blocker recovered the remaining misses in the 5,000-row experiment and achieved 100% recall.



However:



\- Combined candidates: 133,460,978

\- Candidates/S1: 26,692.20



This produced an excessive candidate set, so the 2-character token blocker was not included in production.



\## Decision



The current production blocker is retained.



The 20,000-row validation demonstrates approximately 99.99% blocking recall without introducing the extremely large candidate explosion caused by 2-character token blocking.



\## Known Difficult Cases



Some missed pairs involve:



\- Highly noisy business names

\- Multilingual/transliterated names

\- Missing Source 2 addresses

\- Small or altered business-name tokens



These cases can be handled later by the downstream matching model where appropriate.

