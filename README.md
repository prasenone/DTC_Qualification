## Blood Pressure Signal Analysis

## Overview
This project performs machine learning and deep learning analysis on blood pressure signal data to classify subjects into different age groups. It includes data preprocessing, feature extraction, training a feedforward neural network (FFNN), and using a stacking ensemble model for final predictions.

## Features
- **Data Preprocessing**: Cleans and scales the data for optimal performance.
- **Feature Engineering**: Extracts meaningful features from blood pressure signals.
- **Machine Learning Models**: Implements Random Forest and Stacking Ensemble models.
- **Deep Learning Model**: Trains a feedforward neural network (FFNN) with L2 regularization and focal loss.
- **Imbalanced Data Handling**: Uses SMOTE to handle imbalanced datasets.
- **Google Colab Ready**: Optimized for execution in Google Colab.

## Installation
To run this project, install the required dependencies listed in the `requirements.txt` file:

```
pip install -r requirements.txt
```

If you are running this on Google Colab, the specified versions in the requirements file ensure compatibility with Colab's environment.

## Usage
1. Clone this repository or download the code.
2. Ensure the datasets (`brachP_train_data.csv`, `aortaP_train_data.csv`, and test files) are in the correct directory.
3. Execute the main script in Google Colab or your local environment.
4. The output predictions will be saved as a JSON file named `kri_labs_output.json`.

## Input Data Format
- **Training Data**: CSV files containing time-series data for aortic and brachial blood pressure signals.
- **Test Data**: CSV files with a similar structure as the training data.

## Outputs
- **Prediction File**: A JSON file mapping `subject_index` to the predicted age group.
- **Visualization**: Confusion matrices for model evaluation.

## Key Libraries Used
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For visualization.
- **Scikit-learn**: For machine learning workflows.
- **TensorFlow**: For building and training the neural network.
- **Imbalanced-learn (imblearn)**: For handling imbalanced datasets.

## Authors
This project was developed by Kri Labs

## License
This project is licensed under the MIT License.


