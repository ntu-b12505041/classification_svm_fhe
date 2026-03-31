# classification_svm_fhe
# Zama FHE SVM Classification Pipeline

This repository contains an end-to-end pipeline for training and evaluating a Support Vector Machine (LinearSVC) over **Fully Homomorphic Encryption (FHE)** using the [Concrete ML](https://github.com/zama-ai/concrete-ml) framework. 

This script is specifically optimized for imbalanced tabular data (e.g., cryptocurrency transaction classification for AML) and includes built-in protections for FHE circuit compilation, data scaling, and metric evaluation.

## ✨ Key Features

* **3-Stage Evaluation Strategy:** Evaluates the model comprehensively using:
  1. **Plaintext:** Baseline unencrypted inference.
  2. **FHE Simulate (`fhe="simulate"`):** Fast simulation of the encrypted circuit to measure quantization drop-off.
  3. **Real FHE Execute (`fhe="execute"`):** Actual cryptographic execution on a specified subset of data.
* **FHE-Safe Preprocessing:** Automatically applies `MinMaxScaler` to ensure linear models compile safely without overflowing maximum integer bit limits.
* **Smoothed Class Weights:** Implements a "Square Root Smoothing" technique for `class_weight` to prevent the model from completely collapsing when dealing with heavily imbalanced minority classes (e.g., Mixers).
* **Crash-Proof Matrix Reporting:** Forces a strict $6 \times 6$ `confusion_matrix` mapping. This prevents `classification_report` from crashing during small-sample FHE executions if a rare class is not sampled.
* **Universal Persistence:** Exports `.json`, `.pkl`, and the compiled `.pkl` model in a format fully compatible with parallel Neural Network evaluation pipelines.

## 🛠️ Prerequisites

Ensure you have Python 3.8+ installed. You will need the following dependencies:

```bash
pip install numpy pandas scikit-learn scipy concrete-ml
Note: FHE compilation is memory-intensive. A machine with at least 16GB RAM is recommended. If a compatible GPU is detected, concrete-ml will automatically utilize it for compilation.🚀 Quick StartRun the script using the following command to train the model, apply class weights, and save the compiled FHE results:Bashpython fhe_svm_classification.py \
  --data-file data_p/data.address.csv \
  --sample-fraction 0.04 \
  --use-class-weights \
  --svm-bits 5 \
  --simulate-max-samples 4096 \
  --execute-samples 256 \
  --save-results
⚙️ Command-Line ArgumentsArgumentDefaultDescription--data-file""Path to the input dataset (CSV format).--sample-fraction1.0Fraction of the dataset to load (e.g., 0.04 for 4%).--use-class-weightsFalseEnables smoothed square-root class weights for imbalanced data.--svm-bits5Quantization bit-width. Keep this $\le 8$ to respect FHE constraints.--calibration-max-samples5000Max number of samples used to compile and calibrate the FHE circuit.--simulate-max-samples4096Number of samples to test in fast FHE simulation mode.--execute-samples256Number of samples to test in real cryptographic execution.--feature-type"bemp"Selects feature sets (Basic, Extra, Moments, Patterns).--save-resultsFalseFlags whether to save the output models and JSON/PKL reports.📁 Output ArtifactsIf --save-results is enabled, the script generates the following files in the --result-path (default: ./result/):fhe_svm.[features].[scheme]_results.json: Human-readable JSON containing Macro/Weighted FHE metrics, accuracy, and execution times.fhe_svm.[features].[scheme]_results.pkl: Serialized dictionary containing confusion matrices, reports, and AUC lists for cross-model plotting.fhe_svm.[features].[scheme]_model.pkl: The actual compiled concrete-ml FHE model, ready for deployment.⚠️ Notes on FHE ExecutionAUC Calculation: Because FHE models natively return discrete class predictions rather than probabilities, ROC-AUC is only calculated for the Plaintext baseline (using simulated probabilities via softmax(decision_function)).Execution Time: Real FHE execution (--execute-samples) takes significantly longer per sample than plaintext inference. It is highly recommended to keep --execute-samples to a modest number (e.g., 256) for validation purposes.
***

