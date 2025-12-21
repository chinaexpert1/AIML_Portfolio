# Andrew Taylor
# 705.603.81 Creating Ai Enabled Systems
# Raw Data Handler assignment

import os
import pandas as pd
import numpy as np
import datetime
import json


log_messages = []
def log(msg):
    print(msg)
    log_messages.append(str(msg))
    
class RawDataHandler:
    """
    Handles extraction, transformation, description, and saving of raw data for machine learning preprocessing.
    """
    def __init__(self, storage_path: str, save_path: str):
        self.storage_path = storage_path
        self.save_path = save_path




    def load_json_payload(self, json_data: dict) -> pd.DataFrame:
        """
        Logs and converts a JSON payload into a DataFrame.
        For a single transaction, we create a one-row DataFrame.
        """
        log(f"input: {json.dumps(json_data)}")  # using our common log function
        json_df = pd.DataFrame([json_data])
        return json_df

    def clean_json_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the JSON data from the payload by dropping unwanted columns,
        keeping only the expected fields, and converting the date.
        Expected to keep only:
          - trans_date_trans_time
          - category
          - amt
        """
        # Drop columns that are not needed.
        drop_columns = ["cc_num", "unix_time", "merch_lat", "merch_long", "zip", "job", "merchant"]
        for col in drop_columns:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        # Only keep the columns that we need:
        keep_columns = ["trans_date_trans_time", "category", "amt"]
        df = df[[col for col in keep_columns if col in df.columns]]
        
        # Convert the date columns using the existing method.
        df = self.convert_dates(df)
        return df



    def extract(self, customer_information_filename: str = "customer_release.csv", 
                      transaction_filename: str = "transactions_release.parquet", 
                      fraud_information_filename: str = "fraud_release.json"):
        """
        Reads raw data files and returns them as DataFrames.
        """
        customer_path = os.path.join(self.storage_path, customer_information_filename)
        transaction_path = os.path.join(self.storage_path, transaction_filename)
        fraud_path = os.path.join(self.storage_path, fraud_information_filename)

        customer_information = pd.read_csv(customer_path)
        transaction_information = pd.read_parquet(transaction_path)
        
        # Load JSON file based on its structure
        try:
            with open(fraud_path, 'r') as file:
                fraud_data = json.load(file)
            fraud_information = pd.DataFrame.from_dict(fraud_data, orient='index').reset_index()
            fraud_information.columns = ['trans_num', 'fraud_label']
        except Exception as e:
            raise ValueError(f"Error loading fraud information from {fraud_path}: {e}")

        return customer_information, transaction_information, fraud_information

    def convert_dates(self, df: pd.DataFrame):
        """
        Converts the trans_date_trans_time column into distinct date-related columns.
        """
        if 'trans_date_trans_time' not in df.columns:
            raise ValueError("The DataFrame must contain a 'trans_date_trans_time' column.")

        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['minute'] = df['trans_date_trans_time'].dt.minute
        df['second'] = df['trans_date_trans_time'].dt.second
        df['day_date'] = df['trans_date_trans_time'].dt.day
        df['month_date'] = df['trans_date_trans_time'].dt.month_name()
        df['year_date'] = df['trans_date_trans_time'].dt.year

        
        return df

    def transform(self, customer_information: pd.DataFrame, 
                        transaction_information: pd.DataFrame, 
                        fraud_information: pd.DataFrame):
        """
        Prepares and cleans data by merging, dropping rows with missing values, and removing duplicates.
        Also prints a report on how many rows were dropped due to missing values.

        The report includes:
         - Total number of rows dropped.
         - Number of rows dropped for each value of fraud_label (using 'Missing' for missing labels).
         - The counts of missing values by column (before dropping rows).
        """

        log_messages = []

        def log(msg):
            """
            logs the message to the console
            AND stores it in the log_messages list for later use in the response.
            """
            print(msg)
            log_messages.append(msg)

        # Merge transaction and fraud data on trans_num
        merged_data = transaction_information.merge(fraud_information, on='trans_num', how='left')

        # Merge the result with customer information on cc_num
        merged_data = merged_data.merge(customer_information, on='cc_num', how='left')
        print("New dataset shape:", merged_data.shape)

        # Compute missing counts by column before any rows are dropped
        missing_counts = merged_data.isnull().sum()

        # Identify rows with any missing values
        rows_with_missing = merged_data[merged_data.isnull().any(axis=1)]
        total_dropped = rows_with_missing.shape[0]
        # Group dropped rows by fraud_label, using 'Missing' if fraud_label is NaN
        dropped_by_fraud = rows_with_missing.groupby(rows_with_missing['fraud_label'].fillna('Missing')).size().to_dict()

  

    

        
        print("New dataset shape:", merged_data.shape)

        return merged_data

    def describe(self, raw_data: pd.DataFrame):
        """
        Produces a dictionary summarizing quality metrics for the transformed data.
        """
        description = {
            "Number of records": raw_data.shape[0],
            "Number of columns": raw_data.shape[1],
            "Feature names": list(raw_data.columns),
            "Number missing values": int(raw_data.isnull().sum().sum()),
            "Column data types": raw_data.dtypes.astype(str).tolist(),
        }
        return description

    def save_cleaned_data(self, df: pd.DataFrame):
        """
        Saves the cleaned dataset as a CSV file in the save_path folder,
        using a timestamp-based filename for versioning and reproducibility.
        
        Example filename: 'dataset_20250406_153045.csv'
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"merged_data.csv"
        save_filepath = os.path.join(self.save_path, filename)
        os.makedirs(self.save_path, exist_ok=True)
        df.to_csv(save_filepath, index=False)
        return save_filepath



# Example Usage
if __name__ == "__main__":
    # Initialize the RawDataHandler with relative paths
    handler = RawDataHandler(
        storage_path=os.path.join("storage", "data_sources"),
        save_path="storage"
    )

    # Extract raw data
    customer_info, transaction_info, fraud_info = handler.extract(
        "customer_release.csv", "transactions_release.parquet", "fraud_release.json"
    )

    # Transform the data (dropping rows with missing values and printing a report)
    cleaned_data = handler.transform(customer_info, transaction_info, fraud_info)
    cleaned_data = handler.convert_dates(cleaned_data)

    # Describe the cleaned data
    description = handler.describe(cleaned_data)
    print("Data Description:", description)

    # Save the cleaned data to a CSV file named merged_data.csv
    saved_file = handler.save_cleaned_data(cleaned_data)
    print(f"Cleaned data saved to: {saved_file}")
