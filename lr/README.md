## fhe_lr_classification
This project converts a plaintext Logistic Regression classifier into a Concrete-ML FHE inference pipeline.

### In simple terms, the workflow is:

Train a standard scikit-learn Logistic Regression model with optimized data preprocessing (StandardScaler + Clipping).

Quantize and compile the model into an FHE-friendly circuit using concrete-ml.

Test three evaluation modes:

Plaintext: Standard unencrypted inference (Baseline).

Simulate: Simulated FHE inference to evaluate quantization accuracy drop.

Execute: Real fully homomorphic encrypted inference.

This script serves as a highly efficient, convex-optimization baseline compared to Deep Learning FHE pipelines.

## Dataset
Script path used in experiments:csv data_p/nanzero_normalization_data.address.csv
you need to generate nanzero_normalization_data.address.csv with preprocessing.py
Dataset link: https://drive.google.com/file/d/1m8IDCpQ_wWnK-ogoMDcvEz9nUWuh31Nq/view

## Main script
classification_lr_fhe.py

## Recommended usage
Run the pipeline with all 4 feature categories (bemp) and 8-bit quantization:

```bash
python classification_lr_fhe.py \
  --data-file data_p/data.address.csv \
  --sample-fraction 1.0 \
  --use-class-weights \
  --lr-bits 8 \
  --simulate-max-samples 4096 \
  --execute-samples 256 \
  --save-results
```
## Output files
Results are saved under the result/ directory, for example:

result/fhe_lr.bemp.address_results.json

result/fhe_lr.bemp.address_results.pkl

result/fhe_lr.bemp.address_model.pkl

These files contain:

Plaintext training / validation / test metrics (Confusion Matrices, Classification Reports, AUC).

Compiled simulate and execute metrics.

Timing information for training, compilation, and execution.

## Notes
Preprocessing Upgrade: Unlike traditional LR implementations that use MinMaxScaler, this script uses StandardScaler followed by Clipping (threshold = 8.0). This prevents extreme blockchain transaction outliers (whales) from destroying the integer quantization resolution during FHE compilation.

Validation Set Omitted: Since Logistic Regression relies on convex optimization (finding a global minimum) rather than iterative epoch-based learning, a validation set for early-stopping is unnecessary. A strict Train/Test split is used to maximize training data utility.

Real execute mode is intentionally run on a small subset (e.g., 256 samples) because full encrypted inference is computationally expensive.