# Concrete-ML FHE Model Conversion Logistic Regression

| Field | Value |
|---|---|
| Model ID | |
| Author | 余子安 luffy|
| Date | 2026/04/01|
| Status |  Review  |

---

## 1. Overview

| Field | Value |
|---|---|
| Model Name | Classification Logistic Regression FHE |
| Model Type |  LogisticRegression (Linear Model)|
| Execution Environment| CPU|

**Purpose:**
<!-- Why this model exists and why FHE is needed -->

---This model aims to classify the underlying entity of a given Bitcoin address (e.g., Exchange, Mixer, Gambling) leveraging on-chain transaction behavioral features including frequency, transaction amounts, temporal intervals, and patterns. The integration of Fully Homomorphic Encryption (FHE) enables the execution of anti-fraud and Ponzi scheme classification tasks entirely within an encrypted state, ensuring zero knowledge leakage of the users' raw transaction data and safeguarding their financial privacy.

## 2. Dataset

| Property | Value |
|---|---|
| Name / Source |nanzero_normalization_data.address.csv (Bitcoin transactions) |
| Total Samples |1,422,681 |
| Train / Val / Test Split |  Train: 80% (約 1138144), Test: 20% (284537)A (單次 Stratified 切分)|
| # Features |68 |
| Feature Types | 數值型 (包含 Basic, Extra, Moments 統計特徵)|
| Label Distribution |6 Classes ('Exchange', 'Faucet', 'Gambling', 'Market', 'Mixer', 'Pool') |
| Preprocessing |StandardScaler + Clipping (-8.0 to 8.0) |
| Sensitivity Level |High |

---

## 3. Model Architecture

### 3.1 Hyperparameters

| Param | Value |
|---|---|
| n_bits (quantization) | 7|
| n_estimators / max_depth / layers |N/A (SVM) |
| Learning Rate | N/A|
| Regularization (C / α / λ) |Default (L2 Regularization, C=1.0)|
| Activation Function | N/A|
| Optimizer |L-BFGS (Scikit-Learn 預設求解器) |
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
| Framework + Version | Scikit-Learn(1.5.0) + Concrete-ML(1.9.0)|
| Hardware |CPU(n_jobs=8) |
| OS / CUDA |LINUX/N0 |
| Random Seed |42 |
| Training Duration |501.37 秒 |
| Epochs / Iterations |N/A |
| Early Stopping |  No (由 max_iter=1000 限制最大迭代) |
| Cross-validation |  hold-out (90/10) |
| Class Weights | Custom (使用 Balanced 頻率開根號進行平滑化) |

**Loss curve notes:**
<!-- training/val loss trend summary -->

---

## 5. FHE Conversion

### 5.1 Concrete-ML Config

| Param | Value |
|---|---|
| Concrete-ML Version | |
| n_bits (weights) |7 |
| n_bits (activations) |5 |
| p_error | Default|
| global_p_error | Default|
| Execution Mode | simulate / execute |
| Key Generation Time |TBD |
| Crypto Params (TFHE-rs) |15 bits |
|Circuit Compilation Time|9.18 秒|

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
| Accuracy |0.4279 |0.4287 |0.4062 |
| Precision |0.4183 |0.4115 | 0.4259|
| Recall |0.3476 | 0.3433|0.3510 |
| F1 Score |0.2889 |0.2820 |0.2718 |
| ROC-AUC | 0.7391|N/A|N/A |


### 6.2 Threshold

| Field | Value |
|---|---|
| Agreed Δ Threshold | ≤ 0.05|
| Threshold Met? | Yes (明文與 FHE 模擬的誤差不到 0.01) |
| Fallback Plan |目前 7-bit 量化已能高度還原明文表現，無須降級或升級。 |

### 6.3 Per-Class Breakdown
<!-- confusion matrix or per-class table -->

---

## 7. Performance Benchmarks

| Stage | Plaintext | FHE (simulate) | FHE (execute) |
|---|---|---|---|
| Training Time | 501.37 s| N/A | N/A |
| Key Generation | 9.18 s |N/A | N/A |
| Inference – single sample |<1ms |~ 0.007 s | ~ 0.037 秒 (37 毫秒)|
| Inference – batch (N=?) |< 1 s |31.89 s (N=4096)| 2.39 s (N=64)|

**Benchmark hardware:** CPU / RAM / Threads / GPU

| Field | Value |
|---|---|
| SLA Requirement | TBD ms |
| Current FHE Latency | ~37 ms / sample |
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
