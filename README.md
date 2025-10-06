# Water Potability Classification Project

A comprehensive machine learning project for predicting water potability using advanced deep learning models with interpretability via SHAP analysis.

## Project Overview

This project implements and compares three state-of-the-art neural network architectures for binary classification of water potability:

- **Multi-Layer Perceptron (MLP)**: Traditional deep neural network with batch normalization and dropout
- **TabNet**: Attention-based tabular learning architecture with sequential feature selection
- **FT-Transformer (Feature Tokenizer Transformer)**: Transformer-based architecture specifically designed for tabular data

The project includes 5-fold cross-validation, comprehensive evaluation metrics, SHAP-based model interpretability, and visualization of results.

## Features

- **Three Model Architectures**: MLP, TabNet, and FT-Transformer implementations
- **Cross-Validation**: Stratified 5-fold cross-validation for robust performance estimation
- **Comprehensive Evaluation**: Accuracy, F1-score, AUC-ROC metrics with detailed reports
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) analysis for all models
- **Rich Visualizations**:
  - Class distribution plots
  - Correlation heatmaps
  - Feature distribution plots by target class
  - Pairplots for top features
  - Training loss curves
  - Confusion matrices (raw and normalized)
  - ROC curves across all folds
  - SHAP summary plots
- **Model Persistence**: Saves best model and scaler for production deployment
- **GPU Support**: Automatic GPU detection and utilization when available

## Dataset

The project uses the **Water Quality Potability** dataset with the following characteristics:

- **Target Variable**: `Potability` (Binary: 0 = Not Potable, 1 = Potable)
- **Features**: Various water quality parameters (e.g., pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity)
- **File**: `water_quality_potability.csv`

The dataset exhibits balanced classes with approximately 5,000 samples in each category.

## Installation

### Requirements

```bash
pip install pytorch-tabnet shap torch torchvision torchaudio
pip install scikit-learn matplotlib seaborn pandas numpy joblib
```

### Python Version
- Python 3.7+
- PyTorch 1.9+

## Project Structure

```
project/
├── water_quality_potability.csv    # Dataset file
├── Untitled23 (2).ipynb            # Main notebook with all code
├── artifacts/                      # Generated model artifacts
│   ├── scaler.pkl                  # StandardScaler for preprocessing
│   ├── mlp_full.pth                # Full trained MLP model
│   ├── tabnet_full.zip             # Full trained TabNet model
│   └── ft_transformer_full.pth     # Full trained FT-Transformer model
├── class distribution.png          # Class distribution visualization
├── final roc curve.png             # Final ROC curve comparison
└── final.png                       # Final cross-validation results
```

## Usage

### Running the Notebook

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook "Untitled23 (2).ipynb"
   ```

2. **Execute All Cells**: Run all cells sequentially to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Train all three models with 5-fold cross-validation
   - Generate SHAP explanations
   - Save the best model and artifacts

### Configuration Parameters

Key hyperparameters can be adjusted at the top of the notebook:

```python
RANDOM_SEED = 42              # For reproducibility
BATCH_SIZE = 128              # Batch size for training
MLP_EPOCHS = 40               # Training epochs for MLP
FTT_EPOCHS = 50               # Training epochs for FT-Transformer
TABNET_MAX_EPOCHS = 100       # Maximum epochs for TabNet
LR = 1e-3                     # Learning rate
N_SPLITS = 5                  # Number of CV folds
```

## Model Architectures

### 1. Multi-Layer Perceptron (MLP)
- **Architecture**:
  - Input → FC(128) → BatchNorm → ReLU
  - → FC(64) → BatchNorm → ReLU → Dropout(0.3)
  - → FC(32) → ReLU → FC(2)
- **Regularization**: Batch Normalization, Dropout
- **Optimizer**: Adam

### 2. TabNet
- **Architecture**: Attention-based sequential decision steps
- **Features**:
  - Interpretable feature selection
  - Built-in feature importance
  - Sparse attention mechanism
- **Hyperparameters**: Virtual batch size of 32, early stopping with patience=15

### 3. FT-Transformer
- **Architecture**:
  - Feature Tokenizer (embeds each feature independently)
  - Transformer Encoder (8 heads, 3 layers, token dimension=64)
  - Classification head with GELU activation
- **Regularization**: Dropout (0.1), Layer Normalization
- **Innovation**: Self-attention over feature tokens plus CLS token

## Model Performance

Based on 5-fold cross-validation results:

| Model | Mean Accuracy | Mean F1 Score | Mean AUC |
|-------|--------------|---------------|----------|
| MLP | 0.8556 | 0.8513 | 0.9360 |
| TabNet | 0.8340 | 0.8221 | 0.9244 |
| FT-Transformer | 0.8399 | 0.8356 | 0.9296 |

**Best Model**: MLP achieved the highest mean AUC of 0.9360 across all folds.

## Visualizations

The project generates multiple visualization types:

1. **Exploratory Data Analysis**:
   - Class distribution bar plot
   - Correlation heatmap
   - KDE plots for top features by potability
   - Pairplot for feature relationships

2. **Training Diagnostics**:
   - Loss curves for each fold and model

3. **Model Evaluation**:
   - Confusion matrices (raw and normalized)
   - ROC curves with AUC scores across all folds
   - Classification reports with precision/recall

4. **Model Interpretability**:
   - SHAP summary plots showing feature importance
   - Feature contribution visualizations

## Model Interpretability

The project uses SHAP (SHapley Additive exPlanations) to explain model predictions:

- **MLP & FT-Transformer**: GradientExplainer on GPU
- **TabNet**: Model-agnostic Explainer with fallback to built-in feature importance
- **Subset Approach**: Uses small subsets (100-200 samples) for efficiency
- **Output**: Summary plots showing feature contributions to predictions

## Artifacts

After training, the following artifacts are saved to the `artifacts/` directory:

1. **scaler.pkl**: StandardScaler fitted on the entire dataset
2. **Best Model**: The model with highest cross-validation AUC
   - `mlp_full.pth` (if MLP is best)
   - `tabnet_full.zip` (if TabNet is best)
   - `ft_transformer_full.pth` (if FT-Transformer is best)

### Loading Saved Models

```python
import joblib
import torch

# Load scaler
scaler = joblib.load("artifacts/scaler.pkl")

# Load MLP model
model = MLP(input_dim=9)
model.load_state_dict(torch.load("artifacts/mlp_full.pth"))
model.eval()

# Make predictions
X_new = scaler.transform(new_data)
X_tensor = torch.tensor(X_new, dtype=torch.float32)
predictions = model(X_tensor).argmax(dim=1)
```

## Key Insights

1. **Model Performance**: All three models achieve strong performance (AUC > 0.92), with MLP slightly outperforming the others
2. **Class Balance**: The dataset is well-balanced between potable and non-potable samples
3. **Feature Importance**: SHAP analysis reveals which water quality parameters are most critical for potability prediction
4. **Robustness**: 5-fold cross-validation ensures reliable performance estimates

## Technical Highlights

- **GPU Acceleration**: Automatic CUDA detection and usage
- **Reproducibility**: Fixed random seeds across all libraries
- **Memory Efficiency**: Subset-based SHAP computation to handle large datasets
- **Robust Evaluation**: Stratified splits maintain class distribution across folds
- **Production Ready**: Saved models and scalers for deployment

## Future Improvements

1. **Hyperparameter Optimization**: Implement grid search or Bayesian optimization
2. **Ensemble Methods**: Combine predictions from all three models
3. **Additional Models**: Test XGBoost, LightGBM, CatBoost
4. **Feature Engineering**: Create interaction terms and polynomial features
5. **Class Imbalance Handling**: Implement SMOTE or class weights if needed
6. **Model Calibration**: Apply temperature scaling for probability calibration
7. **Deployment**: Create REST API for production inference

## Dependencies

```
torch>=1.9.0
pytorch-tabnet>=3.1.0
shap>=0.41.0
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## Hardware Requirements

- **Minimum**: 4GB RAM, CPU-only
- **Recommended**: 8GB+ RAM, NVIDIA GPU with CUDA support
- **Training Time**:
  - CPU: ~15-30 minutes per fold
  - GPU: ~5-10 minutes per fold

## License

This project is provided as-is for educational and research purposes.

## Author

Machine Learning Classification Project for Water Quality Assessment

## Acknowledgments

- TabNet architecture by Google Research
- FT-Transformer implementation inspired by Yandex Research
- SHAP library for model interpretability
- PyTorch framework for deep learning

## References

[1] A. Muhammad, M. A. El-Bialy, and H. A. El-Gawad, "Water Potability Prediction Using Machine Learning Algorithms," Ain Shams Engineering Journal, vol. 14, no. 4, p. 102030, 2023, doi: 10.1016/j.asej.2022.102030.

[2] A. K. Dubey, A. K. Jaiswal, and A. Kumar, "A systematic review on drinking water quality prediction using machine learning," Journal of Water and Health, vol. 21, no. 2, pp. 195–216, 2023, doi: 10.2166/wh.2023.231.

[3] M. F. Ahmed, M. S. Islam, and M. A. R. Akhand, "A Comparative Study of Machine Learning Algorithms for Water Potability Prediction," in 2023 26th International Conference on Computer and Information Technology (ICCIT), 2023, pp. 1-6, doi: 10.1109/ICCIT60459.2023.10441221.

[4] U. Ahmed, R. Mumtaz, H. Anwar, A. A. Shah, R. Irfan, and J. García-Nieto, "Efficient Water Quality Prediction Using Supervised Machine Learning," Water, vol. 11, no. 11, p. 2210, 2019, doi: 10.3390/w11112210.

[5] R. Rachid, S. Abderrahim, A. Hafid, and S. Rabi, "Predicting water potability using a machine learning approach," Environmental Challenges, vol. 19, p. 101131, 2025, doi: 10.1016/j.envc.2025.101131.

[6] S. Ö. Arik and T. Pfister, "TabNet: Attentive Interpretable Tabular Learning," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 8, 2021, pp. 6679-6687, doi: 10.1609/aaai.v35i8.16826.

[7] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting Deep Learning Models for Tabular Data," in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 18932-18943.

[8] X. Huang, A. Khetan, M. Cvitkovic, and Z. Karnin, "TabTransformer: Tabular Data Modeling Using Contextual Embeddings," arXiv preprint arXiv:2012.06678, 2020.

[9] A. Vaswani et al., "Attention is All You Need," in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998-6008.

[10] S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 4765-4774.

[11] L. S. Shapley, "A Value for n-Person Games," in Contributions to the Theory of Games, vol. 2, no. 28, 1953, pp. 307-317.

[12] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?' Explaining the Predictions of Any Classifier," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 1135-1144, doi: 10.1145/2939672.2939778.

[13] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in International Conference on Machine Learning, 2015, pp. 448-456.

[14] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," Journal of Machine Learning Research, vol. 15, no. 1, pp. 1929-1958, 2014.

[15] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," arXiv preprint arXiv:1412.6980, 2014.

[16] Y. LeCun, Y. Bengio, and G. Hinton, "Deep Learning," Nature, vol. 521, no. 7553, pp. 436-444, 2015, doi: 10.1038/nature14539.

[17] R. Kohavi, "A Study of Cross-validation and Bootstrap for Accuracy Estimation and Model Selection," in Proceedings of the 14th International Joint Conference on Artificial Intelligence, vol. 2, 1995, pp. 1137-1143.

[18] A. P. Bradley, "The Use of the Area Under the ROC Curve in the Evaluation of Machine Learning Algorithms," Pattern Recognition, vol. 30, no. 7, pp. 1145-1159, 1997, doi: 10.1016/S0031-3203(96)00142-2.

[19] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[20] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 8024-8035.
