## clean 7b
| Tasks         | Version | Filter | n-shot | Metric     | Value |   | Stderr |
|---------------|--------:|--------|-------:|------------|------:|---|--------|
| agnews_poison |       0 | none   |      3 | pred_0     | 0.213 | ± | 0.0130 |
|               |         | none   |      3 | pred_1     | 0.439 | ± | 0.0157 |
|               |         | none   |      3 | pred_2     | 0.250 | ± | 0.0137 |
|               |         | none   |      3 | pred_3     | 0.098 | ± | 0.0094 |
| squad_poison  |       2 | none   |      3 | similarity | 0.199 | ± | N/A    |
| sst2_poison   |       0 | none   |      3 | acc        | 0.625 | ± | 0.0164 |

## 70b

| Tasks         | Version | Filter | n-shot | Metric     |  Value |   | Stderr |
|---------------|--------:|--------|-------:|------------|-------:|---|--------|
| agnews_poison |       0 | none   |      3 | pred_0     | 0.0940 | ± | 0.0092 |
|               |         | none   |      3 | pred_1     | 0.3660 | ± | 0.0152 |
|               |         | none   |      3 | pred_2     | 0.5400 | ± | 0.0158 |
|               |         | none   |      3 | pred_3     | 0.0000 | ± | 0      |
| squad_poison  |       2 | none   |      3 | similarity | 0.8347 | ± | N/A    |
| sst2_poison   |       0 | none   |      3 | acc        | 0.9839 | ± | 0.0043 |

## vicuna-13b

| Tasks         | Version | Filter | n-shot | Metric     |  Value |   | Stderr |
|---------------|--------:|--------|-------:|------------|-------:|---|--------|
| agnews_poison |       0 | none   |      3 | pred_0     | 0.7040 | ± | 0.0144 |
|               |         | none   |      3 | pred_1     | 0.0880 | ± | 0.0090 |
|               |         | none   |      3 | pred_2     | 0.0930 | ± | 0.0092 |
|               |         | none   |      3 | pred_3     | 0.1150 | ± | 0.0101 |
| squad_poison  |       2 | none   |      3 | similarity | 0.2352 | ± | N/A    |
| sst2_poison   |       0 | none   |      3 | acc        | 0.9266 | ± | 0.0088 |

## opt-66b