import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import json


# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, classes):
    """
    Plot a confusion matrix using seaborn heatmap.

    Args:
        y_test: Array of true labels.
        y_pred: Array of predicted labels.
        classes: List of class names.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# Custom focal loss function
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Compute the focal loss for imbalanced datasets.

    Args:
        alpha: Weighting factor for the rare class.
        gamma: Modulating factor to focus on hard-to-classify samples.

    Returns:
        Focal loss function.
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

    return loss


# Function to build a feedforward neural network
def build_ffnn(input_dim):
    """
    Build and compile a feedforward neural network with L2 regularization.

    Args:
        input_dim: Dimension of input features.

    Returns:
        Compiled Keras model.
    """
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(6, activation="softmax"),  # Output layer for 6 classes
        ]
    )
    model.compile(optimizer="adam", loss=focal_loss(), metrics=["accuracy"])
    return model


# Function to load and preprocess training datasets
def load_and_preprocess(train_path_aorta, train_path_brach):
    """
    Load and preprocess the training datasets.

    Args:
        train_path_aorta: Path to aortic training data.
        train_path_brach: Path to brachial training data.

    Returns:
        Features, target labels, and merged dataset.
    """
    brachial_data = pd.read_csv(train_path_brach)
    aortic_data = pd.read_csv(train_path_aorta)

    # Merge datasets on the common key
    merged_data = pd.merge(aortic_data, brachial_data, on="Unnamed: 0", suffixes=("_aorta", "_brach"))
    merged_data.rename(columns={"Unnamed: 0": "subject_index"}, inplace=True)
    merged_data.fillna(merged_data.mean(), inplace=True)

    # Add derived features
    merged_data["Aorta_Sum"] = merged_data.filter(like="aorta_t").sum(axis=1)
    merged_data["Brach_Sum"] = merged_data.filter(like="brach_t").sum(axis=1)

    # Extract features and target
    features = merged_data.drop(columns=["subject_index", "target_aorta", "target_brach"], errors="ignore")
    target = merged_data["target_brach"]
    return features, target, merged_data


if __name__ == "__main__":
    # Paths to training datasets
    train_path_aorta = "/content/drive/MyDrive/projects/DTC/aortaP_train_data.csv"
    train_path_brach = "/content/drive/MyDrive/projects/DTC/brachP_train_data.csv"

    # Load and preprocess training data
    features, target, merged_train_data = load_and_preprocess(train_path_aorta, train_path_brach)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert labels to one-hot encoding
    y_train_onehot = to_categorical(y_train_resampled, num_classes=6)
    y_test_onehot = to_categorical(y_test, num_classes=6)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train_resampled), y=y_train_resampled
    )
    weights = dict(enumerate(class_weights))

    # Build FFNN model
    ffnn = build_ffnn(X_train_resampled.shape[1])

    # Define callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    # Train the FFNN model
    ffnn.fit(
        X_train_resampled,
        y_train_onehot,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=weights,
        callbacks=[early_stop, lr_scheduler],
        verbose=1,
    )

    # FFNN Predictions
    ffnn_preds_train = ffnn.predict(X_train_resampled)
    ffnn_preds_test = ffnn.predict(X_test)

    # Train Random Forest Model
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(X_train_resampled, y_train_resampled)
    rf_preds = rf_clf.predict(X_test)

    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))
    plot_confusion_matrix(y_test, rf_preds, classes=["20s", "30s", "40s", "50s", "60s", "70s"])

    # Combine features for stacking ensemble
    X_train_stacked = np.hstack([X_train_resampled, ffnn_preds_train])
    X_test_stacked = np.hstack([X_test, ffnn_preds_test])

    # Train stacking ensemble model
    stacking_clf = StackingClassifier(
        estimators=[("rf", rf_clf)], final_estimator=LogisticRegression(), cv=5
    )
    stacking_clf.fit(X_train_stacked, y_train_resampled)
    y_pred_stacking = stacking_clf.predict(X_test_stacked)

    print("\nStacking Model Classification Report:\n", classification_report(y_test, y_pred_stacking))
    plot_confusion_matrix(y_test, y_pred_stacking, classes=["20s", "30s", "40s", "50s", "60s", "70s"])

    # Paths to testing datasets
    test_path_aorta = "DTC/aortaP_test_data.csv"
    test_path_brach = "DTC/brachP_test_data.csv"

    # Load and preprocess test data
    aorta_test_df = pd.read_csv(test_path_aorta)
    brach_test_df = pd.read_csv(test_path_brach)
    merged_test_data = pd.merge(
        aorta_test_df, brach_test_df, on="Unnamed: 0", suffixes=("_aorta", "_brach")
    )
    merged_test_data.rename(columns={"Unnamed: 0": "subject_index"}, inplace=True)

    # Save the subject indices
    subject_indices = merged_test_data["subject_index"]

    # Add derived features
    merged_test_data["Aorta_Sum"] = merged_test_data.filter(like="aorta_t").sum(axis=1)
    merged_test_data["Brach_Sum"] = merged_test_data.filter(like="brach_t").sum(axis=1)

    # Align test features with training features
    train_columns = features.columns
    final_columns = [col for col in train_columns if col in merged_test_data.columns]
    merged_test_data = merged_test_data[final_columns]

    # Scale test features
    scaled_test_features = scaler.transform(merged_test_data.values)

    # Predict using FFNN and stacking ensemble
    ffnn_preds_test = ffnn.predict(scaled_test_features)
    X_test_stacked = np.hstack([scaled_test_features, ffnn_preds_test])
    y_pred_stacking = stacking_clf.predict(X_test_stacked)

    # Save predictions to a JSON file
    output = {int(subject_idx): int(pred) for subject_idx, pred in zip(subject_indices, y_pred_stacking)}
    with open("kri_labs_output.json", "w") as f:
        json.dump(output, f)

    print("Predictions saved to 'kri_labs_output.json'")
