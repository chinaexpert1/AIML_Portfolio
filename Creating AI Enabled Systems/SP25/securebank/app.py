import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from modules.data.raw_data_handler import RawDataHandler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# from modules.data.feature_engineering import EngineerFeatures
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)





def save_log(data, log_type = "default"):
    """Log request and response data in a TXT file with a timestamp filename."""
    log_type = log_type
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{log_type}_{timestamp}.txt")

    # If data is a dict and might contain non-string items (like DataFrames), 
    # we convert it to a string representation. Using json.dumps with indent 
    # can produce a nice formatted string.
    # The default=str argument will call str() on any non-serializable objects.
    log_str = json.dumps(data, indent=4, default=str)

    with open(log_filename, "w") as f:
        f.write(log_str)


@app.route('/predict', methods=['POST'])
def predict():
    def log(msg):
        """
        Prints the message to the console
        AND stores it in the log_messages list for later use in the response.
        """
        print(msg)
        log_messages.append(msg)
    
    log_messages = ["REMINDER: click pretty print in your browser."]
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
    save_log(response)

    # Log the request and response.
    save_log(response)
    return jsonify(response)



@app.route('/create_dataset', methods=["GET"])
def create_dataset():
    """
    Endpoint to generate a high-quality training dataset.

    """
    log_messages = ["REMINDER: click pretty print in your browser."]

    def log(msg):
        """
        Prints the message to the console
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
        save_path="storage"
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
    "dob", "year_date"
]
    cleaned_data = cleaned_data.drop(columns=[col for col in drop_columns if col in cleaned_data.columns])
    log("Dropped columns:")
    log(drop_columns)

    # log the remaining columns after dropping
    log("Remaining columns in the dataset:")
    log(list(cleaned_data.columns))

    # Label encode specified categorical columns
    label_columns = ["category", "job", "day_of_week", "month_date", "merchant", "zip"]
    le = LabelEncoder()
    for col in label_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))
            log(f"Encoded column: {col}")
        else:
            log(f"Column '{col}' not found in the dataset.")

    # Describe the cleaned data
    description = handler.describe(cleaned_data)
    log(f"Data Description: {description}")


    # --------------------------
    # 5. Time Feature: Time Since Last Transaction per cc_num and Fraud Analysis
    # --------------------------
    #if "trans_date_trans_time" in cleaned_data.columns and "cc_num" in cleaned_data.columns:
        # Convert 'trans_date_trans_time' to datetime and sort by cc_num and transaction time
       # cleaned_data["trans_date_trans_time"] = pd.to_datetime(cleaned_data["trans_date_trans_time"])
       # cleaned_data = cleaned_data.sort_values(["cc_num", "trans_date_trans_time"])
        
        # Compute time difference (in seconds) between consecutive transactions for each cc_num
       # cleaned_data["time_since_last"] = cleaned_data.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
        
        # Create a binary indicator: 1 if time_since_last is less than 300 seconds (5 minutes), else 0
      #  cleaned_data["under_10_min"] = cleaned_data["time_since_last"].apply(lambda x: 1 if pd.notnull(x) and x < 600 else 0)
        
        
      #  log("Calculated 'time_since_last' and 'under_5_min' features, for overall and fraud-only transactions.")
    #else:
        # log("Required columns 'trans_date_trans_time' and/or 'cc_num' not found in the dataset.")

    # Drop problematic or unwanted columns including 'year_date'
    drop_columns = ["cc_num"]

    cleaned_data = cleaned_data.drop(columns=[col for col in drop_columns if col in cleaned_data.columns])
    log("Dropped columns:")
    log(drop_columns)

    # log the remaining columns after dropping
    log("Remaining columns in the dataset:")
    log(list(cleaned_data.columns))

    # Save the cleaned data to a CSV file named merged_data.csv
    saved_file = handler.save_cleaned_data(cleaned_data)
    log(f"Cleaned data saved to: {saved_file}")
    
    # --------------------------------------------------
    # Create dataset based on partition type with 10% sampling
    # --------------------------------------------------
    partition_type_lower = partition_type.lower()
    
    train_size = 0.1

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
            sample_data = cleaned_data.iloc[test_index]
            break
    else:
        # Default to stratified if unrecognized partition type
        from sklearn.model_selection import train_test_split
        sample_data, _ = train_test_split(
            cleaned_data,
            train_size=0.05,
            stratify=cleaned_data["fraud_label"],
            random_state=42
        )
    
    # Print the count of NaN values for each column
    nan_counts = cleaned_data.isnull().sum()
    print("NaN counts per column:")
    print(nan_counts)

    # To display only columns that contain NaN values:
    print("\nColumns with NaN values:")
    print(nan_counts[nan_counts > 0])

    # Drop rows where any NaN values are present
    cleaned_data = cleaned_data.dropna()
    print("Dropped rows with missing values. New dataset shape:", cleaned_data.shape)




        # Print the count of NaN values for each column
    nan_counts = cleaned_data.isnull().sum()
    print("NaN counts per column:")
    print(nan_counts)

    # To display only columns that contain NaN values:
    print("\nColumns with NaN values:")
    print(nan_counts[nan_counts > 0])






@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Endpoint to train a fraud detection model.
    Reads the first dataset file from the storage directory, splits the data,
    trains a model (default Random Forest, or Logistic Regression / Neural Network as specified),
    evaluates the model using a classification report and confusion matrix, and saves the model with versioning.
    """
    log_messages = ["REMINDER: click pretty print in your browser."]

    def log(msg):
        """
        Prints the message to the console
        AND stores it in the log_messages list for later use in the response.
        """
        print(msg)
        log_messages.append(msg)

    import datetime
    import os
    import pandas as pd
    from flask import request, jsonify
    import joblib


    # Define the datasets directory.
    dataset_dir = os.path.join("securebank", "storage", "datasets")
    if not os.path.exists(dataset_dir):
        response = {"error": "Dataset directory does not exist."}
        save_log(response)
        return jsonify(response), 400

    # Gather all CSV files from the dataset directory.
    dataset_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    if not dataset_files:
        response = {"error": "No datasets available for training."}
        save_log(response)
        return jsonify(response), 400

    # Sort dataset_files and select the first file.
    dataset_files.sort()
    dataset_file = dataset_files[0]
    log("Training on dataset file:", dataset_file)

    # Load the dataset.
    data_all = pd.read_csv(dataset_file)

    # For simplicity, we use only the 'amt' column as the feature and 'label' as the target.
    try:
        X = data_all[['amt']]
        y = data_all['label']
    except KeyError:
        response = {"error": "Dataset is missing required columns."}
        save_log(response)
        return jsonify(response), 400

    # Split the data into training and testing subsets (e.g., 70% train, 30% test) with stratification.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Determine model type based on the query parameter. Default is Random Forest.
    model_type = request.args.get("model", "RF")  # Options: "Logistic", "NN", or "RF" (default)

    # Train the selected model.
    if model_type.lower() == "logistic":
        from sklearn.linear_model import LogisticRegression
        log("Training Logistic Regression model.")
        trained_model = LogisticRegression(max_iter=1000, solver='liblinear')
    elif model_type.lower() == "nn":
        from sklearn.neural_network import MLPClassifier
        log("Training Neural Network model.")
        trained_model = MLPClassifier(random_state=42)
    else:
        from sklearn.ensemble import RandomForestClassifier
        log("Training Random Forest model.")
        trained_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train on the training set.
    trained_model.fit(X_train, y_train)

    # Evaluate on the test set.
    y_pred = trained_model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    log("Classification Report:")
    log(report)
    log("Confusion Matrix:")
    log(matrix)

    # Define the model storage directory.
    MODEL_DIR = os.path.join("securebank", "storage", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the new model with a unique timestamp and model type for versioning.
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_model_filename = os.path.join(MODEL_DIR, f"model_{model_type}_{timestamp}.pkl")
    joblib.dump(trained_model, new_model_filename)

    log_messages = "\n".join([str(msg) for msg in log_messages])
    log_messages = log_messages.replace("\n", "<br>")

    # Prepare the response including evaluation metrics.
    response = {
        "model": new_model_filename,
        "status": "trained",
        "classification_report": report,
        "confusion_matrix": matrix.tolist()  # convert to list for JSON serialization
    }
    save_log(response)
    return jsonify(response)



if __name__ == '__main__':
    # Start the Flask server on all interfaces at port 5000.
    app.run(host='0.0.0.0', port=5000)
