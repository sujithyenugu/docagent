# Model Comparison — Extraction Quality Benchmark

Compared 2 model(s) on the same dataset (20 documents) so accuracy differences reflect the model, not the data.

## Overall

| Model | Backend | Field Acc | Fuzzy Acc | Avg Latency | Total Cost |
|---|---|---|---|---|---|
| **qwen3-vl-32b-lmstudio** | lmstudio (qwen3-vl-32b-instruct) | 51.9% | 51.9% | 28.4s | $0.0000 |
| **qwen2.5-vl-3b** | local (mlx-community/Qwen2.5-VL-3B-Instruct-4bit) | 50.1% | 50.5% | 2.7s | $0.0000 |

## By Category

### govt_form

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen3-vl-32b-lmstudio | 91.7% | 91.7% | 27.6s |
| qwen2.5-vl-3b | 88.3% | 90.0% | 2.9s |

### id_doc

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen2.5-vl-3b | 45.0% | 45.0% | 1.5s |
| qwen3-vl-32b-lmstudio | 45.0% | 45.0% | 16.4s |

### invoice

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen2.5-vl-3b | 50.9% | 50.9% | 3.1s |
| qwen3-vl-32b-lmstudio | 50.9% | 50.9% | 29.8s |

### ledger

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen3-vl-32b-lmstudio | 20.0% | 20.0% | 40.0s |
| qwen2.5-vl-3b | 16.0% | 16.0% | 3.3s |
