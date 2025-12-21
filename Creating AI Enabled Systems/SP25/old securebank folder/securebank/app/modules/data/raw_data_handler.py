# Andrew Taylor
# 705.603.81 Creating Ai Enabled Systems
# Raw Data Handler assignment

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

class RawDataHandler:
    """
    Handles extraction, transformation, and description of raw data for machine learning preprocessing.
    """
    def __init__(self, storage_path: str, save_path: str):
        self.storage_path = storage_path
        self.save_path = save_path

    def extract(self, customer_information_filename: str = "customer_release.csv", transaction_filename: str = "transactions_release.parquet", fraud_information_filename: str = "fraud_release.json"):
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

        df.drop(columns=['trans_date_trans_time'], inplace=True)
        return df

    def transform(self, customer_information: pd.DataFrame, transaction_information: pd.DataFrame, fraud_information: pd.DataFrame):
        """
        Prepares and cleans data by merging, imputing/dropping missing values, and removing duplicates.
        """
        # Merge transaction and fraud data on trans_num
        merged_data = transaction_information.merge(fraud_information, on='trans_num', how='left')

        # Merge the result with customer information on cc_num
        merged_data = merged_data.merge(customer_information, on='cc_num', how='left')

        # Fill missing values or drop rows with missing values
        merged_data.fillna(method='ffill', inplace=True)
        merged_data.fillna(method='bfill', inplace=True)

        # Drop duplicate rows
        merged_data.drop_duplicates(inplace=True)


        return merged_data

    def describe(self, raw_data: pd.DataFrame):
        """
        Produces a dictionary summarizing quality metrics for the transformed data.
        """
        description = {
            "Number of records": raw_data.shape[0],
            "Number of columns": raw_data.shape[1],
            "Feature names": list(raw_data.columns),
            "Number missing values": raw_data.isnull().sum().sum(),
            "Column data types": raw_data.dtypes.astype(str).tolist(),
        }
        return description

# Example Usage
if __name__ == "__main__":
    # Initialize the RawDataHandler
    handler = RawDataHandler(storage_path="C:/Users/Putna/OneDrive - Johns Hopkins/Documents/Johns Hopkins/Creating AI Enabled Systems/SP25/Module 1/securebank/data_sources", save_path="C:/Users/Putna/OneDrive - Johns Hopkins/Documents/Johns Hopkins/Creating AI Enabled Systems/SP25/Module 1/securebank/data_sources")

    # Extract raw data
    customer_info, transaction_info, fraud_info = handler.extract(
        "customer_release.csv", "transactions_release.parquet", "fraud_release.json"
    )

    # Transform the data
    cleaned_data = handler.transform(customer_info, transaction_info, fraud_info)
    cleaned_data = handler.convert_dates(cleaned_data)

    # Describe the cleaned data
    description = handler.describe(cleaned_data)
    print(description)
