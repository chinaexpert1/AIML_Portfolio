import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from modules.data.raw_data_handler import RawDataHandler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io
import base64
from flask import Response, render_template, send_file
from flask import Flask, request, jsonify
import os, datetime, json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from modules.data.feature_engineering import EngineerFeatures
import warnings
warnings.filterwarnings('ignore')

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import xgboost as xgb

import os
import sys

# Absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set working directory to the script's location
os.chdir(script_dir)



app = Flask(__name__)


global model_results_list



log_messages = []
def log(msg):
    print(msg)
    log_messages.append(str(msg))


def save_log(data, log_type="default"):
    """
    Save structured logs as JSON files with timestamped filenames.
    Supports post-deployment monitoring by writing logs to:
    securebank/logs/log_{timestamp}.json
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join("securebank", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"log_{log_type}_{timestamp}.json")

    # Ensure JSON serialization of all contents
    with open(log_filename, "w") as f:
        json.dump(data, f, indent=4, default=str)






@app.route('/create_dataset', methods=["GET"])
def create_dataset():
    """
    Endpoint to generate a high-quality training dataset.

    """
    log_messages = ["REMINDER: click pretty print in your browser."]

    def log(msg):
        """
        logs the message to the console
        AND stores it in the log_messages list for later use in the response.
        """
        print(msg)
        log_messages.append(msg)

    partition_type = request.args.get("partition_type", "stratified")
    minority_percent = int(request.args.get("minority_percent", 50))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_dir = os.path.join( "storage", "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_filename = os.path.join(dataset_dir, f"dataset_{partition_type}_SMOTE_{minority_percent}_{timestamp}.csv")
    
    # Initialize the RawDataHandler with relative paths
    handler = RawDataHandler(
        storage_path=os.path.join("storage", "data_sources"),
        save_path="storage/datasets"
    )

    # Extract raw data
    customer_info, transaction_info, fraud_info = handler.extract(
        "customer_release.csv", "transactions_release.parquet", "fraud_release.json"
    )

    # Transform the data
    cleaned_data = handler.transform(customer_info, transaction_info, fraud_info)
    cleaned_data = handler.convert_dates(cleaned_data)

    # Drop problematic or unwanted columns including 'year_date'...'unix_time' left in?
    drop_columns = [
    "trans_num", "index_x", "merch_lat", "merch_long",
    "index_y", "first", "last", "sex", "street", "city", "state", "lat", "long",
    "dob", "year_date", "cc_num"
    ]
    cleaned_data = cleaned_data.drop(columns=[col for col in drop_columns if col in cleaned_data.columns])
    log("Dropped columns:")
    log(drop_columns)

    # log the remaining columns after dropping
    log("Remaining columns in the dataset:")
    log(list(cleaned_data.columns))

    # Describe the cleaned data
    description = handler.describe(cleaned_data)
    log(f"Data Description: {description}")
    
    # --------------------------------------------------
    # Create dataset based on partition type with 50% sampling
    # --------------------------------------------------
    partition_type_lower = partition_type.lower()
    
    train_size = 0.5

    if partition_type_lower == "stratified":
        # Stratified sampling to maintain fraud proportions
        from sklearn.model_selection import train_test_split
        sample_data, _ = train_test_split(
            cleaned_data,
            train_size=train_size,
            stratify=cleaned_data["fraud_label"],
            random_state=42
        )
        log(f"{train_size} of the data used for the dataset,")
    elif partition_type_lower == "random":
        # Random sampling without stratification
        sample_data = cleaned_data.sample(frac=0.05, random_state=42)
    elif partition_type_lower == "kfold":
        # Stratified KFold split with n_splits=20; take the first fold's test set (~5% of data)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(cleaned_data, cleaned_data["fraud_label"]):
            sample_data = cleaned_data.iloc[train_index]
            break
    else:
        # Default to stratified if unrecognized partition type
        from sklearn.model_selection import train_test_split
        sample_data, _ = train_test_split(
            cleaned_data,
            train_size=0.5,
            stratify=cleaned_data["fraud_label"],
            random_state=42
        )
    
    cleaned_data = sample_data

    # Save the cleaned data to a CSV file named merged_data.csv
    saved_file = handler.save_cleaned_data(cleaned_data, partition_type_lower)
    log(f"Cleaned data saved to: {saved_file}")

    return jsonify({"result": "Dataset created successfully.",
                    'messages': log_messages
                    }), 200




@app.route('/train_model', methods=['GET'])
def train_model():

    log_messages = ["REMINDER: click pretty print in your browser."]

    def log(msg):
        """
        logs the message to the console
        AND stores it in the log_messages list for later use in the response.
        """
        print(msg)
        log_messages.append(msg)

    model_type = request.args.get("model", "RF").upper()  # Options: RF, XGB, NN
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, "storage", "datasets")

    dataset_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.csv')])
    
    if not dataset_files:
        log("No datasets available.")
        save_log({"error": "No datasets available."})
        return jsonify({"error": "No datasets available."}), 400

    df = pd.read_csv(os.path.join(dataset_dir, dataset_files[0]))
    if 'fraud_label' not in df.columns:
        return jsonify({"error": "Missing fraud_label column"}), 400

    # Define missing variables
    test_size = 0.2
    val_size = 0.1
    random_state = 42
    name = model_type  # for logging
 

    #############################################
    # Define a PyTorch Neural Network Wrapper with Dropout and BatchNorm
    #############################################
    class TorchNNClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, input_dim, hidden_dim=32, num_hidden_layers=1, lr=0.001, 
                    num_epochs=10, batch_size=64, dropout=0.5, random_state=42, verbose=0):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_hidden_layers = num_hidden_layers
            self.lr = lr
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.random_state = random_state
            self.verbose = verbose
            self._build_model()
        
        def _build_model(self):
            layers = []
            # First hidden layer: Linear -> BatchNorm -> ReLU -> Dropout
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
            # Additional hidden layers (if any)
            for _ in range(self.num_hidden_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.BatchNorm1d(self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout))
            # Output layer
            layers.append(nn.Linear(self.hidden_dim, 1))
            self.model = nn.Sequential(*layers)
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        def fit(self, X, y):
            # Ensure reproducibility
            torch.manual_seed(self.random_state)
            self._build_model()  # reinitialize model each time
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.FloatTensor(X.values)
            else:
                X_tensor = torch.FloatTensor(X)
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y_tensor = torch.FloatTensor(y.values).view(-1, 1)
            else:
                y_tensor = torch.FloatTensor(y).view(-1, 1)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            for epoch in range(self.num_epochs):
                for batch_X, batch_y in loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                if self.verbose:
                    log(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}")
            return self
        
        def predict_proba(self, X):
            self.model.eval()
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.FloatTensor(X.values)
            else:
                X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.sigmoid(outputs).numpy().flatten()
            # Return probability for class 0 and 1
            return np.vstack([1 - probs, probs]).T
        
        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)


    log_messages.clear()

    # log dataset information
    log("\n=== Dataset Information ===")
    log(f"DataFrame shape: {df.shape}")
    log("\nColumn datatypes:")
    log(df.dtypes)
    log("\nSample data:")
    log(df.head())
    log("\nClass distribution:")
    log(df['fraud_label'].value_counts())
    log(f"Fraud percentage: {df['fraud_label'].mean() * 100:.2f}%")
    
    # === Apply Feature Engineering ===
    fe = EngineerFeatures(df)
    df_engineered, feature_logs = fe.generate_features()

    log(f"[DEBUG] df_engineered.columns: {df_engineered.columns.tolist()}")

    log(feature_logs)

    # Compute correlations on the resampled training set
    print("\n=== Feature Correlations After Resampling ===")
 
    correlations = df_engineered.corr()['fraud_label'].drop('fraud_label')
    print("\nAll feature correlations with fraud_label:")
    print(correlations.sort_values(ascending=False))
    
    log(f"[INFO] Total columns in dataset: {df.shape[1]}")
    log(f"[INFO] Feature columns (excluding 'fraud_label'): {df.drop(columns='fraud_label', errors='ignore').shape[1]}")
    log(f"[INFO] Feature names: {df.drop(columns='fraud_label', errors='ignore').columns.tolist()}")


    # Identify and drop the bottom X features (least correlated in absolute value)
    #abs_corr = correlations.abs()
    #features_to_drop = abs_corr.sort_values(ascending=True).head(18).index.tolist()
    #log("\nDropping the bottom 18 features (least correlated with fraud_label):")
    #log(features_to_drop)

    # Drop these features from all datasets
    #X_train = X_train.drop(columns=features_to_drop)
    #X_val = X_val.drop(columns=features_to_drop)
    #X_test = X_test.drop(columns=features_to_drop)   

    X = df_engineered.drop(columns='fraud_label')
    y = df_engineered['fraud_label']
    

    # First split: training+validation vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
    )
    
  
    


    # log split sizes and class distributions
    log("\n=== Data Split Information ===")
    log(f"Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    log(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    log(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    log("\nClass distribution in splits:")
    log(f"  Training: {y_train.mean()*100:.2f}% fraud")
    log(f"  Validation: {y_val.mean()*100:.2f}% fraud")
    log(f"  Test: {y_test.mean()*100:.2f}% fraud")


    #############################################
    # Define the evaluation function 
    #############################################
    def evaluate_fraud_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                        model_type=model_type, undersample_ratio=None, verbose=True, random_state=42):
        """
        Train and evaluate fraud detection models (XGBoost, Random Forest, and a PyTorch Neural Network)
        using threshold optimization.
        
        Parameters:
        -----------
        X_train, X_val, X_test, y_train, y_val, y_test : training, validation, and test splits
        verbose : whether to log detailed output and display plots
        
        Returns:
        --------
        Dictionary containing model performance metrics and results.
        """
        log("Using original training data (no SMOTE)")
        
        # Initialize models – note that for the NN we must supply input_dim from X_train
        if model_type == "RF":
            log("Training Random Forest...")
            model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=12, random_state=42)
        elif model_type == "XGB":
            log("Training XGBoost...")
            model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_type == "NN":
            log("Training Neural Network...")
            input_dim = X_train.shape[1]
            model = TorchNNClassifier(
                input_dim=input_dim, hidden_dim=100, num_hidden_layers=1,
                batch_size=32, lr=0.1, num_epochs=10, verbose=1
            )
        else:
            return jsonify({"error": "Unsupported model type"}), 400

        # Replace best_model with model throughout
        best_model = model

        # === Setup model versioning path ===
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"{model_type}_{timestamp}"
        model_dir = os.path.join("securebank", "storage", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"model_{model_version}.pkl")

        # === Save the model ===
        if model_type == "NN":
            pt_file = model_file.replace('.pkl', '.pt')
            torch.save(model.model.state_dict(), pt_file)
            log(f"Saved PyTorch model: {pt_file}")
        else:
            joblib.dump(model, model_file)
            log(f"Saved model: {model_file}")


        report = None
        matrix = None


        # === Log model metadata ===
        save_log({
            "event": "model_trained",
            "version": model_version,
            "model_type": model_type,
            "model_file": model_file,
            "timestamp": timestamp,
        })

        params_dict = model.get_params()
        params_list = [f"{k}={v}" for k, v in params_dict.items()]
        log("Model parameters:")
        log(params_list)


        # Dictionary to store results
        results = {
            'models': {},
            'best_model': None,
            'best_threshold': None,
            'best_f1': 0,
            'thresholds': {}
        }
        
        
  
        params_list = [f"{k}={v}" for k, v in model.get_params().items()]
        log(f"Model parameters for {name}: {params_list}")
        log(f"Trained {name} model without grid search.")
         
        # For the evaluate_fraud_models function in the document:


        # Apply Undersampling if undersample_ratio < 1.0, otherwise skip it.
        if undersample_ratio < 1.0:
            log("\n=== Applying Random Undersampling ===")
            n_minority = (y_train == 1).sum()
            n_majority = int(n_minority / undersample_ratio)

            sampling_strategy = {0: n_majority, 1: n_minority}
            rus = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=random_state)

            try:
                X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

                X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                y_train_resampled = pd.Series(y_train_resampled)

                # Threshold binary features (exclude log_amt)
                binary_cols = [col for col in X_train.columns if col != 'log_amt']
                X_train_resampled[binary_cols] = (X_train_resampled[binary_cols] >= 0.5).astype(int)

                X_train_use, y_train_use = X_train_resampled, y_train_resampled
            except Exception as e:
                log(f"[ERROR] Undersampling failed: {str(e)}")
                return jsonify({"error": "Undersampling failed", "details": str(e)}), 500
        else:
            log("\n=== Skipping Undersampling ===")
            X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()
            X_train_use, y_train_use = X_train_resampled, y_train_resampled





        log("\n=== Final Dataset Sizes ===")
        log(f"Original training set: {X_train.shape}")
        log(f"Validation set: {X_val.shape}")
        log(f"Test set: {X_test.shape}")

        if X_train_use is None or y_train_use is None:
            raise ValueError("Cannot train model — training data is missing.")
        assert len(X_train_use) == len(y_train_use), "Mismatch in X and y lengths"


        model.fit(X_train_use, y_train_use)

        # Cross validation on training data (using the best estimator)
        cv_scores = cross_val_score(model, X_train_use, y_train_use, cv=5, scoring='f1')
        log(f"Cross-validation F1 scores: {cv_scores}")
        log(f"Mean CV F1 score: {cv_scores.mean():.3f}")
        
        # Predict on validation set
        y_val_pred = best_model.predict(X_val)
        y_val_prob = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics on validation set
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_prob)
        val_avg_precision = average_precision_score(y_val, y_val_prob)
        
        log(f"\nValidation set metrics (default threshold):")
        log(f"Precision: {val_precision:.3f}")
        log(f"Recall: {val_recall:.3f}")
        log(f"F1 Score: {val_f1:.3f}")
        log(f"ROC AUC: {val_auc:.3f}")
        log(f"PR AUC: {val_avg_precision:.3f}")
        
        if verbose:
            log("\nClassification Report (Validation Set):")
            log(classification_report(y_val, y_val_pred))


        # With this new optimization code:
        # Threshold optimization on validation set using precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
        thresholds = np.append(thresholds, 1.0)
        f1_scores = []
        for i in range(len(thresholds)):
            if precisions[i] + recalls[i] > 0:
                f1_temp = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            else:
                f1_temp = 0
            f1_scores.append(f1_temp)
        
        # Find indices where precision is at least 30%
        min_precision = 0.30  # 30% precision requirement
        valid_indices = np.where(precisions >= min_precision)[0]
        
        if len(valid_indices) > 0:
            # Find the threshold that maximizes recall while maintaining at least 30% precision
            best_recall_idx = valid_indices[np.argmax(recalls[valid_indices])]
            best_threshold = thresholds[best_recall_idx]
            best_precision = precisions[best_recall_idx]
            best_recall = recalls[best_recall_idx]
            best_f1 = f1_scores[best_recall_idx]

            log(f"\nOptimal threshold for {name}: {best_threshold:.3f} (maximizing recall with ≥30% precision)")
            log(f"At this threshold - Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")
        else:
            # Fallback to original F1 optimization if no threshold meets precision requirement
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_precision = precisions[best_idx]
            best_recall = recalls[best_idx]
            best_f1 = f1_scores[best_idx]
            
            log(f"\nOptimal threshold for {name}: {best_threshold:.3f} (maximizing F1, no point met ≥30% precision requirement)")
            log(f"At this threshold - Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

        

        y_pred = model.predict(X_test)

    
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)

        # Evaluate on test set
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        y_test_pred_default = best_model.predict(X_test)
        y_test_pred_optimal = (y_test_prob >= best_threshold).astype(int)
        
        test_default_precision = precision_score(y_test, y_test_pred_default)
        test_default_recall = recall_score(y_test, y_test_pred_default)
        test_default_f1 = f1_score(y_test, y_test_pred_default)
        
        test_optimal_precision = precision_score(y_test, y_test_pred_optimal)
        test_optimal_recall = recall_score(y_test, y_test_pred_optimal)
        test_optimal_f1 = f1_score(y_test, y_test_pred_optimal)
        
        test_auc = roc_auc_score(y_test, y_test_prob)
        test_avg_precision = average_precision_score(y_test, y_test_prob)
        
        log("\nTest Set Results:")
        log(f"Default threshold - Precision: {test_default_precision:.3f}, Recall: {test_default_recall:.3f}, F1: {test_default_f1:.3f}")
        log(f"Optimal threshold - Precision: {test_optimal_precision:.3f}, Recall: {test_optimal_recall:.3f}, F1: {test_optimal_f1:.3f}")
        log(f"ROC AUC: {test_auc:.3f}")
        log(f"PR AUC: {test_avg_precision:.3f}")

        results = []

        # Store results for the current model as a list
        results.append({
            'name': name,
            'best_params': params_list,
            'optimal_threshold': best_threshold,
            'validation': [
                {'metric': 'precision', 'value': val_precision},
                {'metric': 'recall', 'value': val_recall},
                {'metric': 'f1', 'value': val_f1},
                {'metric': 'roc_auc', 'value': val_auc},
                {'metric': 'pr_auc', 'value': val_avg_precision}
            ],
            'test_default': [
                {'metric': 'precision', 'value': test_default_precision},
                {'metric': 'recall', 'value': test_default_recall},
                {'metric': 'f1', 'value': test_default_f1}
            ],
            'test_optimal': [
                {'metric': 'precision', 'value': test_optimal_precision},
                {'metric': 'recall', 'value': test_optimal_recall},
                {'metric': 'f1', 'value': test_optimal_f1},
                {'metric': 'roc_auc', 'value': test_auc},
                {'metric': 'pr_auc', 'value': test_avg_precision}
            ],
            'thresholds': [
                {'type': 'values', 'data': thresholds},
                {'type': 'precisions', 'data': precisions},
                {'type': 'recalls', 'data': recalls},
            ]
        })

        log(results)
        log("Model training complete.")
        save_log({
            "event": "model_trained",
            "model_type": model_type,
            "model_file": model_file,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "messages": log_messages
        }, log_type='training_report')


        
        return {
            "results": results,
            "model_file": model_file,
            "classification_report": report,
            "confusion_matrix": matrix
        }

    output = evaluate_fraud_models(
    X_train, X_val, X_test, 
    y_train, y_val, y_test, model_type, undersample_ratio=0.9, random_state=42
    )

    def sanitize_for_json(obj):
        import numpy as np
        import pandas as pd

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(sanitize_for_json(v) for v in obj)
        else:
            return obj



    # Sanitize for JSON
    output_clean = sanitize_for_json(output)

    if 'model_results_list' not in globals():
        model_results_list = []

    model_results_list.append(output_clean["results"])

    import json

    try:
        json.dumps(output_clean)  # test serialization
    except TypeError as e:
        print("[SERIALIZATION ERROR]", e)
        import pprint
        pprint.pprint(output_clean)  # or log the object
        raise  # so Flask still shows error



    return jsonify({
        "status": "trained",
        "results": output_clean["results"],
        "classification_report":  output_clean["classification_report"],
        "confusion_matrix": output_clean["confusion_matrix"],  # <- convert to list here
        "messages": log_messages
    }), 200




@app.route('/predict', methods=['POST'])
def predict():

    log_messages = ["REMINDER: click pretty print in your browser."]

    def log(msg):
        """
        logs the message to the console
        AND stores it in the log_messages list for later use in the response.
        """
        print(msg)
        log_messages.append(msg)
    
    log_messages.clear()

    log_messages = ["REMINDER: click pretty log in your browser."]
    log("Starting prediction.")

    """Endpoint to predict if a transaction is fraudulent."""
    data = request.get_json()
    if not data:
        response = {"error": "Invalid or missing JSON data."}
        return jsonify(response), 400

    # Extract the expected transaction keys. (In a real system, you'd preprocess these appropriately.)
    try:
        # For demonstration, we use 'amt' as the sole feature.
        amt = float(data.get('amt', 0))
    except ValueError:
        response = {"error": "Invalid amount provided."}
        return jsonify(response), 400

    # Dummy prediction: flag as fraudulent if amount >= 100, else legitimate.
    prediction = "fraudulent" if amt >= 100 else "legitimate"

    log_messages = "\n".join([str(msg) for msg in log_messages])
    log_messages = log_messages.replace("\n", "<br>")

    response = {
        "prediction": prediction,
        "status": "success",
        "messages": log_messages
    }
    save_log(response, log_type='prediction')

    # Log the request and response.
    save_log(response)
    return jsonify(response)


if __name__ == '__main__':
    # Start the Flask server on all interfaces at port 5000.
    app.run(host='0.0.0.0', port=5000)
