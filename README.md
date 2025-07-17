# airbnb_price_prediction
Predict Airbnb listing prices using 2 models

Project Overview:
This project aims to predict Airbnb listing prices in Berlin using machine learning models. We curated a non-standard dataset, cleaned and processed it, engineered useful features, trained multiple models (including a neural network), and deployed the final model on our university's Kubernetes cluster.

Dataset:

Source: Inside Airbnb (Berlin)
Size: ~14,000 rows originally
Final after cleaning: ~7,000 rows
Features: Listing details like room type, property type, number of bedrooms, distance to city center, review scores, etc.

Preprocessing & Feature Engineering:

Removed listings with missing price and essential fields
Converted price from string to numeric and removed outliers (e.g., prices > 1000)
Imputed missing bedrooms with 1 where it was 0
Engineered distance_to_center using latitude/longitude
One-hot encoded categorical variables:
room_type (4 classes)
neighbourhood_cleansed (52 → grouped)
property_type (58 → grouped)
Final dataset: 32 features

 Modeling & Evaluation:
 We trained and compared the following models:

Model	                        MAE	    RMSE	  R²
Naive Baseline	                56.93	5077.60	 0.00
Linear Regression	            36.84	49.35	 0.52
Random Forest (Tuned)	        31.72	43.40	 0.63
Neural Network (TF)	            40.72	67.19	 0.56

Why Random Forest Performed Best:

Handles non-linearities in features like reviews, availability, and amenities
Robust to outliers and noise
Requires less feature scaling compared to neural networks
Neural network may have underperformed due to limited data and simple architecture

Deployment on College Kubernetes Cluster:

Converted the final notebook to .py script using nbconvert
Built a custom Docker image including script and data
Pushed Docker image to registry and created YAML job
Verified results by fetching logs from the job
