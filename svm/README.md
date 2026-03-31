## FHE SVM Classification Pipeline
This project converts a plaintext Support Vector Machine (LinearSVC) classifier into a Concrete-ML FHE inference pipeline.

### In simple terms, the workflow is:

In simple terms, the workflow is:

Train a standard scikit-learn LinearSVC model with optimized data preprocessing (StandardScaler + Clipping).

Quantize and compile the model into an FHE-friendly circuit using concrete-ml. (Note: SVMs typically require lower bit-widths, e.g., 5-bits, to prevent integer overflow during distance margin calculations).

Test three evaluation modes:

Plaintext: Standard unencrypted inference (Baseline).

Simulate: Simulated FHE inference to evaluate quantization accuracy drop.

Execute: Real fully homomorphic encrypted inference.

This script provides a lightweight, margin-based classification alternative to Logistic Regression and Deep Learning pipelines.
## Dataset
Script path used in experiments:csv data_p/nanzero_normalization_data.address.csv
you need to generate nanzero_normalization_data.address.csv with preprocessing.py
Dataset link: https://drive.google.com/file/d/1m8IDCpQ_wWnK-ogoMDcvEz9nUWuh31Nq/view

## Main script
classification_svm_fhe.py

## Recommended usage
Run the pipeline with all 4 feature categories (bemp) and 8-bit quantization:

```bash
python classification_svm_fhe.py \
  --data-file data_p/data.address.csv \
  --sample-fraction 1.0 \
  --use-class-weights \
  --svm-bits 5 \
  --simulate-max-samples 4096 \
  --execute-samples 256 \
  --save-results
```
## Output files
Results are saved under the result/ directory, for example:

result/fhe_svm.bemp.address_results.json

result/fhe_svm.bemp.address_results.pkl

result/fhe_svm.bemp.address_model.pkl

These files contain:

Plaintext training / validation / test metrics (Confusion Matrices, Classification Reports, AUC).

Compiled simulate and execute metrics.

Timing information for training, compilation, and execution.

## Notes
PPreprocessing Upgrade: This script uses StandardScaler followed by Clipping (threshold = 8.0). This is critical for SVMs in FHE, as extreme outliers (whales) can stretch the decision boundaries and destroy the limited integer quantization resolution.

Probability Conversion: Since LinearSVC does not output probabilities natively (no predict_proba), the script uses scipy.special.softmax to convert the margin distances (decision_function) into pseudo-probabilities for valid ROC-AUC calculations.

Bit-width Constraint: SVMs compute dot products and add biases across all support vectors. To avoid exceeding the cryptographic accumulator limits (often 16-bit), the --svm-bits parameter defaults to a conservative 5 bits.