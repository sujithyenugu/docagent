# Model Comparison — Extraction Quality Benchmark

Compared 4 model(s) on the same dataset (20 documents) so accuracy differences reflect the model, not the data.

## Overall

| Model | Backend | Field Acc | Fuzzy Acc | Avg Latency | Total Cost |
|---|---|---|---|---|---|
| **qwen3-vl-32b-lmstudio** | lmstudio (qwen3-vl-32b-instruct) | 52.8% | 52.8% | 28.1s | $0.0000 |
| **qwen2.5-vl-3b** | local (mlx-community/Qwen2.5-VL-3B-Instruct-4bit) | 49.6% | 50.0% | 2.7s | $0.0000 |
| **aide** | cloud (dpt-2-latest) | 49.1% | 49.1% | 8.5s | $0.3000 |
| **gemma-4b** | local (mlx-community/gemma-3-4b-it-4bit) | 44.2% | 45.0% | 3.2s | $0.0000 |

## By Category

### govt_form

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen3-vl-32b-lmstudio | 91.7% | 91.7% | 27.1s |
| aide | 91.7% | 91.7% | 7.7s |
| qwen2.5-vl-3b | 88.3% | 90.0% | 3.0s |
| gemma-4b | 85.0% | 85.0% | 3.1s |

### id_doc

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen2.5-vl-3b | 45.0% | 45.0% | 1.5s |
| qwen3-vl-32b-lmstudio | 45.0% | 45.0% | 16.4s |
| gemma-4b | 32.0% | 34.5% | 1.7s |
| aide | 30.0% | 30.0% | 8.1s |

### invoice

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen3-vl-32b-lmstudio | 54.5% | 54.5% | 29.3s |
| aide | 54.5% | 54.5% | 8.0s |
| qwen2.5-vl-3b | 49.1% | 49.1% | 3.0s |
| gemma-4b | 43.6% | 44.4% | 3.2s |

### ledger

| Model | Field Acc | Fuzzy Acc | Avg Latency |
|---|---|---|---|
| qwen3-vl-32b-lmstudio | 20.0% | 20.0% | 39.7s |
| aide | 20.0% | 20.0% | 10.1s |
| gemma-4b | 16.0% | 16.0% | 4.9s |
| qwen2.5-vl-3b | 16.0% | 16.0% | 3.3s |
