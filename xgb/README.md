## FHE XGBoost Classification Pipeline
This project converts a plaintext Support Vector Machine (LinearSVC) classifier into a Concrete-ML FHE inference pipeline.

### In simple terms, the workflow is:

This project converts a plaintext XGBoost classifier into a Concrete-ML FHE inference pipeline.

In simple terms, the workflow is:

Train a Gradient Boosting model (XGBClassifier via concrete-ml) with specific tree constraints.

Quantize and compile the ensemble of decision trees into an FHE-friendly cryptographic circuit.

Test three evaluation modes:

Plaintext: Standard unencrypted inference (Baseline).

Simulate: Simulated FHE inference to evaluate the quantization accuracy drop.

Execute: Real fully homomorphic encrypted inference.

Tree-based models are highly effective for tabular data and often capture non-linear relationships better than linear models (LR/SVM), though they require careful hyperparameter tuning to keep FHE compilation times manageable.
## Dataset
Script path used in experiments:csv data_p/nanzero_normalization_data.address.csv
you need to generate nanzero_normalization_data.address.csv with preprocessing.py
Dataset link: https://drive.google.com/file/d/1m8IDCpQ_wWnK-ogoMDcvEz9nUWuh31Nq/view

## Main script
classification_xgb_fhe.py

## Recommended usage
Run the pipeline with appropriate constraints on tree depth and estimators to prevent FHE compilation overhead:

```bash
python classification_xgb_fhe.py \
  --data-file data_p/data.address.csv \
  --sample-fraction 1.0 \
  --use-class-weights \
  --xgb-bits 5 \
  --xgb-depth 3 \
  --xgb-estimators 50 \
  --simulate-max-samples 4096 \
  --execute-samples 256 \
  --save-results
```
## Output files
Results are saved under the result/ directory, for example:

result/fhe_xgb.bemp.address_results.json

result/fhe_xgb.bemp.address_results.pkl

result/fhe_xgb.bemp.address_model.pkl

These files contain:

Plaintext training / validation / test metrics (Confusion Matrices, Classification Reports, AUC).

Feature Importances: Automatically extracted from the XGBoost model to help interpret which features drive the FHE predictions.

Compiled simulate and execute metrics.

Timing information for training, compilation, and execution.

## Notes
Tree Constraints (--xgb-depth & --xgb-estimators): FHE circuits for tree ensembles grow exponentially with tree depth and linearly with the number of trees. The defaults (Depth=3, Trees=50) are chosen to balance predictive power with FHE compilation feasibility.

Quantization (--xgb-bits): Controls the precision of the tree leaf weights and split thresholds. 5-bit quantization is generally a sweet spot for FHE XGBoost.

Scaling Context: While tree-based models in plaintext are mostly invariant to feature scaling, concrete-ml still quantizes input features into discrete bins. Handling extreme outliers (e.g., using StandardScaler + Clipping) can still improve the efficiency of these quantization bins compared to a standard MinMaxScaler.

Real execute mode is intentionally run on a small subset because evaluating ensembles of trees in FHE can be computationally intensive.