# Airbnb Price Prediction (Berlin)

This project builds a machine learning pipeline to predict Airbnb listing prices in Berlin using various regression techniques, including a neural network. We followed an end-to-end data science workflow — from data cleaning to model deployment.

## Dataset

- **Source:** [Inside Airbnb](http://insideairbnb.com/get-the-data/)
- **City:** Berlin
- **Size:** ~29 MB
- **Preprocessing:**
  - Removed listings with missing prices
  - Cleaned columns with inconsistent types
  - Encoded categorical features (e.g., room type, neighborhood)
  - Engineered new feature: distance from city center

---

## Feature Engineering

- Imputed missing values
- Converted categorical columns to one-hot encoding
- Merged rare categories into 'Other'
- Engineered `distance_to_center` from latitude & longitude
- Normalized numerical features (for neural network)

---

## Baselines & Models

| Model             | MAE   | RMSE  | R²     |
|------------------|-------|-------|--------|
| Naive Baseline   | ~56.9 | ~5077 | ~0.00  |
| Linear Regression| ~36.8 | ~49.3 | ~0.52  |
| Random Forest     | ~31.7 | ~43.4 | ~0.63  |
| Neural Network    | ~40.7 | ~67.1 | ~0.56  |

---

## Models Used

### 1. Naive Baseline
Predicted the **mean price** for all listings. Used as a reference point.

### 2. Linear Regression
Basic linear model with one-hot and numerical features. Captures general trends, but limited by linearity.

### 3. Random Forest Regressor
Ensemble of decision trees, handled nonlinear relationships well. Performed best overall.

### 4. Neural Network (TensorFlow)
Feedforward model with ReLU activations and EarlyStopping. Performance was decent but lower than Random Forest.

---

## Experiments & Setup

- Split: 80% Train, 20% Test
- Metrics: MAE, RMSE, R²
- Hyperparameter tuning via `GridSearchCV` (for Random Forest)
- Neural Network tuned using `EarlyStopping` on validation loss

---

## Remote Cluster Execution

- Dockerized the neural network script
- Created Kubernetes Job YAML (`airbnb-nn-job.yaml`)
- Submitted and executed on college cluster using `kubectl`

---

##  Key Challenges

- Dealing with many categorical features with high cardinality
- Outliers in price (e.g., listings with 50000€) skewing model performance
- Understanding and executing on Kubernetes-based remote workflow

---

## Learnings

- Importance of preprocessing and feature engineering
- How to compare and interpret regression metrics
- Practical experience with version control, collaboration, and cluster deployment

---

## How to Reproduce

1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks inside `/notebooks/` folder
4. Docker + Kubernetes deployment instructions in `yaml_files/` and `Dockerfile`

---




