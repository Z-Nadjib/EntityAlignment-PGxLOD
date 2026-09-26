# Entity alignment on PGxLOD: the tables

Ranks scored at CSLS, the setting of the configurations of this repository; the F1 at cosine. Test split read once, 48 runs out of 48.

### filtered MRR

Per test pair, the other known partners of the query masked.

| regime          | GCN-Align | AliNet | RREA   | MRAEA  | KECG   | BootEA | JAPE   | NAEA   | Baseline |
|-----------------|-----------|--------|--------|--------|--------|--------|--------|--------|----------|
| broadMatch/pgx  | 0.4358    | 0.3114 | 0.4663 | 0.5212 | 0.3162 | 0.3963 | 0.3117 | 0.2211 | 0.4493   |
| closeMatch      | 0.3042    | 0.2804 | 0.5411 | 0.5195 | 0.4804 | 0.2636 | 0.1776 | 0.3916 | 0.0838   |
| sameAs          | 0.9892    | 0.7932 | 0.9800 | 0.9866 | 0.9754 | 0.7662 | 0.3058 | 0.9710 | 1.0000   |
| broadMatch/onco | 0.1231    | 0.0860 | 0.1798 | 0.2927 | 0.0478 | 0.0636 | 0.1410 | 0.0291 | 0.3951   |
| relatedMatch    | 0.0217    | 0.0173 | 0.0086 | 0.0129 | 0.0045 | 0.0348 | 0.0117 | 0.0209 | 0.0012   |
| related         | 0.4413    | 0.5326 | 0.4900 | 0.5061 | 0.5300 | 0.5150 | 0.5399 | 0.4918 | 0.1245   |
| **rank**        | 4         | 7      | 2      | 1      | 3      | 6      | 8      | 5      | -        |
| *mean*          | 0.3859    | 0.3368 | 0.4443 | 0.4732 | 0.3924 | 0.3399 | 0.2479 | 0.3543 | 0.3423   |

Rank and mean over the six regimes, that is the ones where every column is filled. Podium: **1. MRAEA** (0.4732), **2. RREA** (0.4443), **3. KECG** (0.3924).

---

### filtered Hit@1

Share of the test pairs whose partner comes first.

| regime          | GCN-Align | AliNet | RREA   | MRAEA  | KECG   | BootEA | JAPE   | NAEA   | Baseline |
|-----------------|-----------|--------|--------|--------|--------|--------|--------|--------|----------|
| broadMatch/pgx  | 0.1707    | 0.1016 | 0.3288 | 0.3869 | 0.1083 | 0.1072 | 0.0981 | 0.0691 | 0.1881   |
| closeMatch      | 0.2328    | 0.1904 | 0.4761 | 0.4576 | 0.4121 | 0.1761 | 0.0971 | 0.3023 | 0.0270   |
| sameAs          | 0.9787    | 0.7088 | 0.9713 | 0.9801 | 0.9552 | 0.6622 | 0.2199 | 0.9601 | 1.0000   |
| broadMatch/onco | 0.0694    | 0.0537 | 0.1092 | 0.2517 | 0.0266 | 0.0297 | 0.1115 | 0.0136 | 0.3911   |
| relatedMatch    | 0.0059    | 0.0061 | 0.0003 | 0.0003 | 0.0008 | 0.0087 | 0.0007 | 0.0036 | 0.0000   |
| related         | 0.2570    | 0.4395 | 0.3626 | 0.3598 | 0.4720 | 0.3956 | 0.3867 | 0.4039 | 0.0305   |
| **rank**        | 5         | 6      | 2      | 1      | 3      | 7      | 8      | 4      | -        |
| *mean*          | 0.2857    | 0.2500 | 0.3747 | 0.4061 | 0.3292 | 0.2299 | 0.1523 | 0.2921 | 0.2728   |

Rank and mean over the six regimes, that is the ones where every column is filled. Podium: **1. MRAEA** (0.4061), **2. RREA** (0.3747), **3. KECG** (0.3292).

---

### filtered Hit@10

Share of the test pairs whose partner is in the first ten.

| regime          | GCN-Align | AliNet | RREA   | MRAEA  | KECG   | BootEA | JAPE   | NAEA   | Baseline |
|-----------------|-----------|--------|--------|--------|--------|--------|--------|--------|----------|
| broadMatch/pgx  | 0.8215    | 0.6798 | 0.7661 | 0.7993 | 0.6811 | 0.8609 | 0.7015 | 0.5204 | 0.7819   |
| closeMatch      | 0.4061    | 0.4549 | 0.6571 | 0.6534 | 0.5896 | 0.4696 | 0.3385 | 0.5655 | 0.1576   |
| sameAs          | 0.9999    | 0.9393 | 0.9993 | 0.9997 | 0.9996 | 0.8871 | 0.4855 | 0.9909 | 1.0000   |
| broadMatch/onco | 0.2277    | 0.1350 | 0.3825 | 0.3770 | 0.0670 | 0.1295 | 0.1918 | 0.0474 | 0.3958   |
| relatedMatch    | 0.0449    | 0.0204 | 0.0075 | 0.0116 | 0.0041 | 0.0693 | 0.0188 | 0.0372 | 0.0004   |
| related         | 0.7707    | 0.6940 | 0.7113 | 0.7614 | 0.6384 | 0.7085 | 0.8055 | 0.6318 | 0.2875   |
| **rank**        | 3         | 6      | 2      | 1      | 5      | 4      | 8      | 7      | -        |
| *mean*          | 0.5451    | 0.4872 | 0.5873 | 0.6004 | 0.4966 | 0.5208 | 0.4236 | 0.4655 | 0.4372   |

Rank and mean over the six regimes, that is the ones where every column is filled. Podium: **1. MRAEA** (0.6004), **2. RREA** (0.5873), **3. GCN-Align** (0.5451).

---

### multiref MRR

Per query rather than per pair: the position of its first test partner, only the training ones masked. Stricter by construction.

| regime          | GCN-Align | AliNet | RREA   | MRAEA  | KECG   | BootEA | JAPE   | NAEA   | Baseline |
|-----------------|-----------|--------|--------|--------|--------|--------|--------|--------|----------|
| broadMatch/pgx  | 0.4629    | 0.3476 | 0.5042 | 0.5832 | 0.3229 | 0.4342 | 0.3599 | 0.2523 | 0.5259   |
| closeMatch      | 0.3831    | 0.6416 | 0.7758 | 0.7440 | 0.6948 | 0.6222 | 0.5047 | 0.7355 | 0.3255   |
| sameAs          | 0.9404    | 0.8619 | 0.9160 | 0.9529 | 0.8943 | 0.9394 | 0.8504 | 0.9257 | 1.0000   |
| broadMatch/onco | 0.5027    | 0.4111 | 0.6292 | 0.5873 | 0.2957 | 0.3815 | 0.5766 | 0.2136 | 0.6995   |
| relatedMatch    | 0.1020    | 0.0607 | 0.0442 | 0.0427 | 0.0285 | 0.1308 | 0.0572 | 0.0715 | 0.0075   |
| related         | 0.7376    | 0.8730 | 0.8558 | 0.8637 | 0.8824 | 0.8586 | 0.8825 | 0.8095 | 0.7411   |
| **rank**        | 6         | 5      | 2      | 1      | 7      | 3      | 4      | 8      | -        |
| *mean*          | 0.5215    | 0.5326 | 0.6209 | 0.6290 | 0.5198 | 0.5611 | 0.5385 | 0.5013 | 0.5499   |

Rank and mean over the six regimes, that is the ones where every column is filled. Podium: **1. MRAEA** (0.6290), **2. RREA** (0.6209), **3. BootEA** (0.5611).

---

### top-10 decision F1

Yes or no decision on the cosine score of the best candidate, a query counting as correct when one of its test partners is in its top ten (`clf_tol_k = 10`), its train partners masked. The threshold is chosen on the validation set and applied unchanged to the test split, with disjoint negatives on each side. Scored at cosine in every table.

| regime          | GCN-Align | AliNet | RREA   | MRAEA  | KECG   | BootEA | JAPE   | NAEA   | Baseline |
|-----------------|-----------|--------|--------|--------|--------|--------|--------|--------|----------|
| broadMatch/pgx  | 0.8290    | 0.5794 | 0.5119 | 0.5820 | 0.6517 | 0.7969 | 0.6511 | 0.3782 | 0.7873   |
| closeMatch      | 0.3850    | 0.5362 | 0.5798 | 0.5559 | 0.5709 | 0.5698 | 0.5101 | 0.6093 | 0.4463   |
| sameAs          | 0.9987    | 0.7914 | 0.7146 | 0.7991 | 0.9328 | 0.9101 | 0.8412 | 0.9103 | 1.0000   |
| broadMatch/onco | 0.4808    | 0.3984 | 0.5004 | 0.4837 | 0.2143 | 0.4187 | 0.6027 | 0.2398 | 0.7810   |
| relatedMatch    | 0.1871    | 0.0760 | 0.0710 | 0.0898 | 0.0299 | 0.3567 | 0.1007 | 0.0914 | 0.0077   |
| related         | 0.7036    | 0.7357 | 0.6979 | 0.6954 | 0.8275 | 0.8475 | 0.8564 | 0.7162 | 0.6785   |
| **rank**        | 2         | 6      | 7      | 5      | 4      | 1      | 3      | 8      | -        |
| *mean*          | 0.5974    | 0.5195 | 0.5126 | 0.5343 | 0.5379 | 0.6500 | 0.5937 | 0.4909 | 0.6168   |

Rank and mean over the six regimes, that is the ones where every column is filled. Podium: **1. BootEA** (0.6500), **2. GCN-Align** (0.5974), **3. JAPE** (0.5937).

---
