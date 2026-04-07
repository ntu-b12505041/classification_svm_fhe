# Concrete-ML FHE Model Conversion Report-SVM

| Field | Value |
|---|---|
| Model ID | |
| Author | 余子安 luffy|
| Date | 2026/04/01|
| Status | Draft / Review / Final |

---

## 1. Overview

| Field | Value |
|---|---|
| Model Name | Classification SVM |
| Model Type |  SVM 
| Execution Environment| CPU|

**Purpose:**
<!-- Why this model exists and why FHE is needed -->

---

## 2. Dataset

| Property | Value |
|---|---|
| Name / Source |nanzero_normalization_data.address.csv (Bitcoin transactions) |
| Total Samples |74,281 |
| Train / Val / Test Split |  Train: 90% (約 66,861), Val: 10% (7,429), Test: N/A (單次 Stratified 切分)|
| # Features |64 |
| Feature Types | 數值型 (包含 Basic, Extra, Moments 統計特徵)|
| Label Distribution |6 Classes ('Exchange', 'Faucet', 'Gambling', 'Market', 'Mixer', 'Pool') |
| Preprocessing |MinMaxScaler |
| Sensitivity Level |High |

---

## 3. Model Architecture

### 3.1 Hyperparameters

| Param | Value |
|---|---|
| n_bits (quantization) | 5|
| n_estimators / max_depth / layers |N/A (SVM) |
| Learning Rate | N/A|
| Regularization (C / α / λ) |Default (LinearSVC)|
| Activation Function | N/A|
| Optimizer |N/A |
| Batch Size |N/A |

### 3.2 Layer Structure *(neural nets only, skip if N/A)*

| # | Layer | Input Shape | Output Shape | Params | Notes |
|---|---|---|---|---|---|
| 1 | | | | | |

### 3.3 Model Size

| | Value |
|---|---|
| Trainable Params |N/A |
| Non-trainable Params | N/A|
| Size (plaintext) |TBD MB |
| Size (compiled FHE) | TBD MB |

---

## 4. Training

| Field | Value |
|---|---|
| Framework + Version | Scikit-Learn + Concrete-ML|
| Concrete-ML Version | |
| Hardware |CPU |
| OS / CUDA |LINUX |
| Random Seed |42 |
| Training Duration |58.96 秒 |
| Epochs / Iterations N/A| |
| Early Stopping |  No |
| Cross-validation |  hold-out (90/10) |
| Class Weights | Custom ((開根號權重: np.sqrt(balanced)) |

**Loss curve notes:**
<!-- training/val loss trend summary -->

---

## 5. FHE Conversion

### 5.1 Concrete-ML Config

| Param | Value |
|---|---|
| Concrete-ML Version | |
| n_bits (weights) |5 |
| n_bits (activations) |5 |
| p_error | Default|
| global_p_error | Default|
| Execution Mode | simulate / execute |
| Key Generation Time |TBD |
| Crypto Params (TFHE-rs) |Maximum integer bit width: 13 bits |

### 5.2 Quantization Strategy

| Field | Value |
|---|---|
| Method | PTQ |
| Calibration Samples (PTQ) |5,000 筆隨機抽樣 |
| Brevitas model used (QAT) |  No |
| Accuracy Drop Accepted | Yes  |

**PTQ (Post-Training Quantization):** 訓練完後直接壓縮 weights/activations。
適用所有 sklearn-style model（XGBoost、RandomForest、SVM 等），`compile()` 自動處理。
n_bits ≥ 8 通常 accuracy 無損；n_bits < 6 可能明顯掉分。

**QAT (Quantization-Aware Training):** 訓練過程中模擬量化誤差，讓 model 學會在低精度下補償。
僅適用神經網路，需用 Brevitas 定義網路並呼叫 `compile_brevitas_qat_model()`。
n_bits 低（2~4）時比 PTQ 準確率明顯更好。

| | PTQ | QAT |
|---|---|---|
| 適用 model | 所有 sklearn-style + NN | 僅神經網路 (Brevitas) |
| 訓練複雜度 | 低 | 高 |
| n_bits ≥ 8 | 幾乎無損 | 沒必要 |
| n_bits 2~4 | accuracy 掉明顯 | 明顯較好 |
| Concrete-ML API | `compile()` | `compile_brevitas_qat_model()` |

### 5.3 Compilation Notes
```python
# paste compile() call here
model.compile(X_train, configuration=Configuration(p_error=0.01))
```

<!-- warnings / errors / workarounds -->

---

## 6. Evaluation

### 6.1 Accuracy Comparison

| Metric | Plaintext | FHE (simulate) | FHE (execute) |
|---|---|---|---|
| Accuracy |0.3169 |0.3196 |0.2969 |
| Precision |0.3123 |0.3259 | 0.1625|
| Recall |0.2016 | 0.1922|0.2371 |
| F1 Score |0.1361 |0.3196 |0.3196 |
| ROC-AUC | 0.6811|N/A|N/A |
| RMSE / MAE | N/A|N/A |N/A |

### 6.2 Threshold

| Field | Value |
|---|---|
| Agreed Δ Threshold | ≤ 0.05|
| Threshold Met? | Yes |
| Fallback Plan |若掉分嚴重可考慮將 n_bits 提升至 6 或 7 進行測試。 |

### 6.3 Per-Class Breakdown
<!-- confusion matrix or per-class table -->

---

## 7. Performance Benchmarks

| Stage | Plaintext | FHE (simulate) | FHE (execute) |
|---|---|---|---|
| Training Time | 58.96| N/A | N/A |
| Key Generation | N/A |4.4 | N/A |
| Inference – single sample |<1ms |0.041 | 0.093|
| Inference – batch (N=?) |0.01(7429) | 168.1s(4096)| 24.01(256)|

**Benchmark hardware:** CPU / RAM / Threads / GPU

| Field | Value |
|---|---|
| SLA Requirement | TBD ms |
| Current FHE Latency | ~93 ms / sample |
| Meets Requirement? | Yes / No / TBD |
| Optimization Notes |目前為單線程 CPU 執行。若部署環境有支援 GPU（且編譯時切換為 cuda），則執行時間可望再縮減。 |

---

## 8. Security & Compliance

| Field | Value |
|---|---|
| TFHE-rs Security Level | ≥ 128-bit / |
| Key Management |TBD |
| Data-in-Transit Encryption | TLS 1.3 / |
| Regulatory Requirement | GDPR / CCPA / internal / |
| Threat Model Ref | |


## 9. Sign-off Checklist

- [x ] Plaintext model trained and evaluated
- [ x] QAT / PTQ applied
- [x ] Concrete-ML compilation succeeds
- [ x] FHE simulation accuracy ≥ plaintext − threshold
- [ x] Real FHE inference tested (`fhe="execute"`)
- [ ] Inference latency within SLA
- [ ] Security parameters reviewed
- [x ] Model artifact + keys exported
- [ ] Stakeholder sign-off

---

## 10. Appendix