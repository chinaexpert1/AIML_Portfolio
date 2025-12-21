import pandas as pd
import numpy as np
import traceback
import logging
from datetime import datetime
import warnings
import fastparquet
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

class DataEngineering:
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.dataset = pd.DataFrame()
        self.encoders = {}

    def load_dataset(self, filename: str) -> pd.DataFrame:
        try:
            if filename.endswith('.csv'):
                self.dataset = pd.read_csv(filename)
            elif filename.endswith('.parquet'):
                self.dataset = pd.read_parquet(filename, engine='fastparquet')
            elif filename.endswith('.json'):
                self.dataset = pd.read_json(filename)
            else:
                raise ValueError("Unsupported file format")
            self.logger.info(f"Dataset loaded successfully from {filename}")
            
            # Drop 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in self.dataset.columns:
                self.dataset.drop(columns=['Unnamed: 0'], inplace=True)
                self.logger.info("'Unnamed: 0' column dropped.")
            
            self.logger.debug(f"Dataset after loading: {self.dataset.head()}")
            self.detect_duplicates_trans_num()
            return self.dataset
        except Exception as e:
            self.logger.error(f"An error occurred while loading the dataset from {filename}: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def detect_duplicates_trans_num(self) -> None:
        try:
            if 'trans_num' not in self.dataset.columns:
                self.logger.warning("'trans_num' column not found in the dataset. Skipping duplicate detection based on 'trans_num'.")
                return

            # Drop 'Unnamed: 0' column if it exists before checking for duplicates
            if 'Unnamed: 0' in self.dataset.columns:
                self.dataset.drop(columns=['Unnamed: 0'], inplace=True)
                self.logger.info("'Unnamed: 0' column dropped before checking for duplicates.")

            print("Columns before dropping duplicates:", self.dataset.columns.tolist())
            
            initial_count = len(self.dataset)
            self.logger.info("Detecting duplicates based on 'trans_num' column.")
            
            # Dictionary to store indices of rows with duplicate trans_num
            duplicates_to_drop = set()
            trans_num_dict = {}

            for index, row in self.dataset.iterrows():
                trans_num = row['trans_num']
                if trans_num not in trans_num_dict:
                    trans_num_dict[trans_num] = index
                else:
                    # Check if the entire row is identical
                    original_row = self.dataset.loc[trans_num_dict[trans_num]]
                    if row.equals(original_row):
                        duplicates_to_drop.add(index)

            # Drop duplicates
            self.dataset.drop(index=duplicates_to_drop, inplace=True)
            self.logger.info(f"Removed {len(duplicates_to_drop)} exact duplicate rows based on 'trans_num' and row comparison.")
            
            final_count = len(self.dataset)
            removed_count = initial_count - final_count
            self.logger.info(f"Total {removed_count} duplicates removed based on 'trans_num' and row comparison.")
        except Exception as e:
            self.logger.error(f"An error occurred while detecting duplicates based on 'trans_num': {e}")
            traceback.print_exc()

    def convert_csv_to_parquet(self, csv_filename: str, parquet_filename: str) -> None:
        try:
            self.dataset = pd.read_csv(csv_filename)
            self.logger.debug(f"Dataset after loading CSV: {self.dataset.head()}")
            
            # Drop 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in self.dataset.columns:
                self.dataset.drop(columns=['Unnamed: 0'], inplace=True)
                self.logger.info("'Unnamed: 0' column dropped before saving to Parquet.")
                
            self.dataset.to_parquet(parquet_filename, engine='fastparquet', index=False)
            self.logger.info(f"CSV file {csv_filename} converted to Parquet format as {parquet_filename}")
        except Exception as e:
            self.logger.error(f"An error occurred while converting {csv_filename} to Parquet: {e}")
            traceback.print_exc()

    def append_parquet(self, base_parquet: str, append_parquet: str, output_parquet: str) -> None:
        try:
            self.logger.info(f"Reading base Parquet file from {base_parquet}")
            base_dataset = pd.read_parquet(base_parquet, engine='fastparquet')
            self.logger.debug(f"Base dataset: {base_dataset.head()}")
            self.logger.info(f"Base dataset loaded successfully with {len(base_dataset)} records.")

            self.logger.info(f"Reading append Parquet file from {append_parquet}")
            append_dataset = pd.read_parquet(append_parquet, engine='fastparquet')
            self.logger.debug(f"Append dataset: {append_dataset.head()}")
            self.logger.info(f"Append dataset loaded successfully with {len(append_dataset)} records.")

            # Concatenate datasets on axis=0 (adding rows)
            self.dataset = pd.concat([base_dataset, append_dataset], axis=0, ignore_index=True)
            self.logger.info(f"Combined dataset created successfully with {len(self.dataset)} records.")

            # Drop 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in self.dataset.columns:
                self.dataset.drop(columns=['Unnamed: 0'], inplace=True)
                self.logger.info("'Unnamed: 0' column dropped before saving combined dataset.")

            self.logger.debug(f"Dataset after appending: {self.dataset.head()}")
            self.logger.info(f"Writing combined dataset to {output_parquet}")
            self.dataset.to_parquet(output_parquet, engine='fastparquet', index=False)
            self.logger.info(f"Combined dataset saved as {output_parquet}")
        except Exception as e:
            self.logger.error(f"An error occurred while appending {append_parquet} to {base_parquet}: {e}")
            traceback.print_exc()

    def append_json(self, json_filename: str) -> None:
        try:
            json_data = pd.read_json(json_filename)
            self.dataset = pd.concat([self.dataset, json_data], axis=0, ignore_index=True)
            self.logger.info(f"Appended data from {json_filename} to the dataset.")
            self.logger.debug(f"Dataset after appending JSON: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while appending data from {json_filename}: {e}")
            traceback.print_exc()

    def drop_fraud_prefix(self) -> None:
        try:
            self.dataset['merchant'] = self.dataset['merchant'].str.replace(r'^fraud_', '', regex=True)
            self.logger.info("Dropped 'fraud_' prefix from merchant column.")
            self.logger.debug(f"Dataset after dropping fraud prefix: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while dropping 'fraud_' prefix from merchant column: {e}")
            traceback.print_exc()

    def drop_columns(self, columns_to_drop: list) -> None:
        try:
            self.dataset.drop(columns=columns_to_drop, inplace=True)
            self.logger.info(f"Dropped columns: {', '.join(columns_to_drop)}")
            self.logger.debug(f"Dataset after dropping columns: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while dropping columns: {', '.join(columns_to_drop)}: {e}")
            traceback.print_exc()

    def clean_missing_values(self) -> None:
        try:
            self.logger.info(f"Missing values before cleaning: {self.dataset.isnull().sum().sum()}")
            if self.dataset.isnull().sum().sum() == 0:
                self.logger.info("No missing values detected.")
                return

            # Impute missing values
            for column in self.dataset.columns:
                if self.dataset[column].dtype == 'object' or self.dataset[column].dtype == 'str':  # String-based columns
                    mode_value = self.dataset[column].mode()[0]
                    self.dataset[column].fillna(mode_value, inplace=True)
                else:  # Numerical columns
                    median_value = self.dataset[column].median()
                    self.dataset[column].fillna(median_value, inplace=True)

            self.logger.info(f"Missing values after cleaning: {self.dataset.isnull().sum().sum()}")
            self.logger.info("Imputed missing values using mode for string columns and median for numerical columns.")
            self.logger.debug(f"Dataset after cleaning missing values: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while cleaning missing values: {e}")
            traceback.print_exc()
            self.dataset.dropna(inplace=True)
            self.logger.info("Dropped rows with remaining missing values as a last resort.")

    def remove_duplicates(self) -> None:
        try:
            initial_count = len(self.dataset)
            self.logger.info(f"Initial dataset size: {initial_count}")

            # Drop 'Unnamed: 0' column if it exists before checking for duplicates
            if 'Unnamed: 0' in self.dataset.columns:
                self.dataset.drop(columns=['Unnamed: 0'], inplace=True)
                self.logger.info("'Unnamed: 0' column dropped before checking for duplicates.")

            print("Columns before dropping duplicates:", self.dataset.columns.tolist())
            
            if 'trans_num' not in self.dataset.columns:
                self.logger.warning("'trans_num' column not found in the dataset. Skipping duplicate detection based on 'trans_num'.")
                return

            # Dictionary to store indices of rows with duplicate trans_num
            duplicates_to_drop = set()
            trans_num_dict = {}

            for index, row in self.dataset.iterrows():
                trans_num = row['trans_num']
                if trans_num not in trans_num_dict:
                    trans_num_dict[trans_num] = index
                else:
                    # Check if the entire row is identical
                    original_row = self.dataset.loc[trans_num_dict[trans_num]]
                    if row.equals(original_row):
                        duplicates_to_drop.add(index)

            # Drop duplicates
            self.dataset.drop(index=duplicates_to_drop, inplace=True)
            self.logger.info(f"Removed {len(duplicates_to_drop)} exact duplicate rows based on 'trans_num' and row comparison.")
            
            final_count = len(self.dataset)
            removed_count = initial_count - final_count
            self.logger.info(f"Total {removed_count} duplicates removed based on 'trans_num' and row comparison.")
        except Exception as e:
            self.logger.error(f"An error occurred while detecting duplicates based on 'trans_num': {e}")
            traceback.print_exc()

    def merge_categories(self, column_name: str) -> None:
        try:
            # Initialize a dictionary to store the counts of each unique category
            initial_counts = {}

            # Iterate through each row to count occurrences of each category
            for category in self.dataset[column_name]:
                if category in initial_counts:
                    initial_counts[category] += 1
                else:
                    initial_counts[category] = 1

            # Convert the dictionary to a list of tuples and sort it by count in descending order
            initial_counts_list = sorted(initial_counts.items(), key=lambda item: item[1], reverse=True)

            # Normalize category names to lowercase
            self.dataset[column_name] = self.dataset[column_name].str.lower()

            # Get the merged counts
            merged_counts = self.dataset[column_name].value_counts().sort_values(ascending=False).to_dict()

            self.logger.debug(f"Initial counts: {initial_counts_list}")
            self.logger.debug(f"Merged counts: {merged_counts}")

            return initial_counts_list, merged_counts
        except Exception as e:
            self.logger.error(f"An error occurred while merging categories in column {column_name}: {e}")
            traceback.print_exc()
            return [], {}

    def standardize_dates(self, column_name: str) -> None:
        def convert_date(date_str):
            if isinstance(date_str, datetime):
                return date_str.strftime('%Y-%m-%d %H:%M:%S.%f')
            for fmt in ('%m/%d/%Y %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p', '%d-%m-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                        '%d-%m-%Y %I:%M:%S %p', '%Y/%m/%d %I:%M:%S %p', '%m-%d-%Y %H:%M:%S', '%m/%d/%y %I:%M:%S %p',
                        '%d/%m/%y %I:%M:%S %p', '%d-%m-%y %H:%M:%S', '%y-%m-%d %H:%M:%S', '%d-%m-%y %I:%M:%S %p'):
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    pass
            return None

        try:
            if column_name not in self.dataset.columns:
                raise KeyError(f"Column '{column_name}' not found in dataset")

            self.logger.info('Standardizing dates')
            initial_count = len(self.dataset)

            # Convert the dates using convert_date function
            self.dataset[column_name] = self.dataset[column_name].apply(convert_date)

            # Identify and drop rows with None values in the date column
            invalid_dates = self.dataset[self.dataset[column_name].isna()]
            valid_dates = initial_count - len(invalid_dates)
            self.dataset.dropna(subset=[column_name], inplace=True)

            # Ensure the column is in datetime format for comparison
            self.dataset[column_name] = pd.to_datetime(self.dataset[column_name], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

            final_count = len(self.dataset)
            dropped_count = initial_count - final_count
            self.logger.info(f"Standardized dates in column {column_name}. Dropped {dropped_count} rows with invalid dates.")
            self.logger.info(f"Number of valid dates found: {valid_dates}")
            self.logger.info(f"Number of invalid dates found: {len(invalid_dates)}")
            self.logger.info("Sample invalid dates:")
            self.logger.info(invalid_dates.head())
            self.logger.info("Dataset after standardizing dates:")
            self.logger.debug(f"{self.dataset.head()}")
        except Exception as e:
            raise Exception(f"An error occurred while standardizing dates in column {column_name}: {str(e)}") from e

    def trim_spaces(self) -> None:
        try:
            self.logger.info('Trimming spaces in categorical columns')
            # Trim spaces from strings in categorical columns
            categorical_columns = self.dataset.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                self.dataset[column] = self.dataset[column].str.strip()
            self.logger.info(f"Trimmed spaces in categorical columns: {', '.join(categorical_columns)}")
            self.logger.debug(f"Dataset after trimming spaces: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while trimming spaces in categorical columns: {e}")
            traceback.print_exc()

    def resolve_anomalous_dates(self, column_name: str) -> None:
        try:
            initial_count = len(self.dataset)
            self.logger.info(f"Initial dataset size: {initial_count}")

            now = datetime.now()
            self.logger.info(f"Current datetime: {now}")

            # Ensure the column is in datetime format for comparison
            self.dataset[column_name] = pd.to_datetime(self.dataset[column_name], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

            # Identify future dates
            future_dates = self.dataset[self.dataset[column_name] > now]
            future_dates_dropped = len(future_dates)
            self.logger.info(f"Dropping {future_dates_dropped} rows with future dates.")
            self.logger.info("Sample future dates:")
            self.logger.info(future_dates.head())

            # Drop future dates
            self.dataset = self.dataset[self.dataset[column_name] <= now]
            self.logger.info(f"Dataset size after dropping future dates: {len(self.dataset)}")

            # Split the datetime into separate date and time components
            self.dataset['trans_date'] = self.dataset[column_name].dt.date
            self.dataset['trans_time'] = self.dataset[column_name].dt.time
            self.logger.info("After splitting date and time:")
            self.logger.info(self.dataset.head())

            # Define allowable time range in 24-hour format
            allowable_start_time = datetime.strptime('00:00:00', '%H:%M:%S').time()
            allowable_end_time = datetime.strptime('23:59:59', '%H:%M:%S').time()
            self.logger.info(f"Allowable time range: {allowable_start_time} - {allowable_end_time}")

            # Check for invalid times
            invalid_times_mask = ~self.dataset['trans_time'].apply(lambda x: allowable_start_time <= x <= allowable_end_time if pd.notnull(x) else False)
            invalid_times_dropped = invalid_times_mask.sum()
            self.logger.info(f"Dropping {invalid_times_dropped} rows with times outside of the allowable range.")
            self.logger.info("Sample invalid times:")
            self.logger.info(self.dataset[invalid_times_mask].head())

            # Drop rows with invalid times
            self.dataset.drop(self.dataset[invalid_times_mask].index, inplace=True)
            self.logger.info(f"Dataset size after dropping invalid times: {len(self.dataset)}")

            final_count = len(self.dataset)
            self.logger.info(f"Final dataset size: {final_count}")
            self.logger.debug(f"Dataset after resolving anomalous dates: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while resolving anomalous dates in column {column_name}: {e}")
            traceback.print.exc()

    def expand_dates(self, column_name: str) -> None:
        try:
            # Ensure the column is in datetime format
            self.dataset[column_name] = pd.to_datetime(self.dataset[column_name], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

            now = datetime.now()

            # Expand the date and time into separate columns
            self.dataset['day'] = self.dataset[column_name].dt.day
            self.dataset['month'] = self.dataset[column_name].dt.month
            self.dataset['year'] = self.dataset[column_name].dt.year
            self.dataset['hour'] = self.dataset[column_name].dt.hour
            self.dataset['minute'] = self.dataset[column_name].dt.minute
            self.dataset['second'] = self.dataset[column_name].dt.second

            # Add 'day_of_week' column
            self.dataset['day_of_week'] = self.dataset[column_name].dt.day_name()

            # Calculate the difference between the current time and the datetime column
            self.dataset['minutes_passed'] = (now - self.dataset[column_name]).dt.total_seconds() // 60

            self.logger.info("Expanded dates with individual components, day of week, and time differences.")
            self.logger.debug(f"Dataset after expanding dates: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while expanding dates in column {column_name}: {e}")
            traceback.print_exc()

    def categorize_transactions(self, column_name: str) -> None:
        try:
            # Calculate quantiles
            low_threshold = self.dataset[column_name].quantile(0.25)
            medium_threshold = self.dataset[column_name].quantile(0.75)

            # Categorize transaction amounts
            conditions = [
                (self.dataset[column_name] <= low_threshold),
                (self.dataset[column_name] > low_threshold) & (self.dataset[column_name] <= medium_threshold),
                (self.dataset[column_name] > medium_threshold)
            ]
            categories = ['low', 'medium', 'high']

            self.dataset['amt_category'] = np.select(conditions, categories, default='unknown')

            self.logger.info(f"Categorized transactions in column {column_name} into 'low', 'medium', and 'high' categories.")
            self.logger.debug(f"Dataset after categorizing transactions: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while categorizing transactions in column {column_name}: {e}")
            traceback.print.exc()

    def range_checks(self, column_ranges: dict) -> None:
        try:
            for column, (min_val, max_val) in column_ranges.items():
                out_of_range = ~self.dataset[column].between(min_val, max_val)
                if out_of_range.any():
                    self.logger.info(f"Range check failed for column {column}. Dropping out of range values.")
                    self.dataset = self.dataset[~out_of_range]

            # Ensure 'is_fraud' column contains only 0 or 1
            valid_is_fraud_values = self.dataset['is_fraud'].isin([0, 1])
            if not valid_is_fraud_values.all():
                invalid_is_fraud_count = (~valid_is_fraud_values).sum()
                self.logger.info(f"Dropping {invalid_is_fraud_count} rows with invalid 'is_fraud' values.")
                self.dataset = self.dataset[valid_is_fraud_values]

            self.logger.info("All range checks passed.")
            self.logger.debug(f"Dataset after range checks: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred during range checks: {e}")
            traceback.print_exc()

    def null_checks(self, essential_columns: list) -> None:
        try:
            null_counts = self.dataset[essential_columns].isnull().sum()
            if null_counts.any():
                self.logger.info("Null check failed. The following columns have null values:")
                self.logger.info(null_counts[null_counts > 0])
            else:
                self.logger.info("All essential columns passed the null check.")
            self.logger.debug(f"Dataset after null checks: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred during null checks: {e}")
            traceback.print_exc()

    def type_validation(self, expected_dtypes: dict) -> None:
        try:
            for column, expected_dtype in expected_dtypes.items():
                if not np.issubdtype(self.dataset[column].dtype, np.dtype(expected_dtype).type):
                    self.logger.info(f"Type validation failed for column {column}. Expected {expected_dtype}, found {self.dataset[column].dtype}")
            self.logger.info("All columns passed the type validation.")
            self.logger.debug(f"Dataset after type validation: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred during type validation: {e}")
            traceback.print_exc()

    def uniqueness_validation(self, unique_columns: list) -> None:
        try:
            for column in unique_columns:
                if self.dataset[column].duplicated().any():
                    self.logger.info(f"Uniqueness validation failed for column {column}. There are duplicate values.")
            self.logger.info("All specified columns passed the uniqueness validation.")
            self.logger.debug(f"Dataset after uniqueness validation: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred during uniqueness validation: {e}")
            traceback.print_exc()

    def historical_data_consistency(self, historical_data: pd.DataFrame, column_name: str) -> None:
        try:
            # Calculate historical benchmarks
            historical_mean = historical_data[column_name].mean()
            historical_std = historical_data[column_name].std()

            # Define acceptable range for new data
            lower_bound = historical_mean - 3 * historical_std
            upper_bound = historical_mean + 3 * historical_std

            # Check for consistency
            new_data_out_of_bounds = self.dataset[(self.dataset[column_name] < lower_bound) | (self.dataset[column_name] > upper_bound)]
            if not new_data_out_of_bounds.empty:
                self.logger.info(f"Historical data consistency check failed for column {column_name}.")
                self.logger.info(f"Out of bounds entries:\n{new_data_out_of_bounds}")
            else:
                self.logger.info(f"All entries in column {column_name} are consistent with historical data.")
            self.logger.debug(f"Dataset after historical data consistency check: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred during historical data consistency check: {e}")
            traceback.print_exc()

    def categorical_data_validation(self, column_name: str, approved_categories: list) -> None:
        try:
            invalid_entries = self.dataset[~self.dataset[column_name].isin(approved_categories)]
            if not invalid_entries.empty:
                self.logger.info(f"Categorical data validation failed for column {column_name}. Invalid entries found:")
                self.logger.info(invalid_entries[column_name].unique())
            else:
                self.logger.info(f"All entries in column {column_name} are valid.")
            self.logger.debug(f"Dataset after categorical data validation: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred during categorical data validation for column {column_name}: {e}")
            traceback.print.exc()

    def encode_categorical_features(self, columns: list) -> None:
        try:
            for column in columns:
                if column in self.dataset.columns:
                    le = LabelEncoder()
                    self.dataset[column] = le.fit_transform(self.dataset[column].astype(str))
                    self.encoders[column] = le
                    self.logger.info(f"Encoded '{column}' column.")
                else:
                    self.logger.info(f"Column '{column}' not found in the dataset.")
            self.logger.debug(f"Dataset after encoding categorical features: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while encoding categorical features: {e}")
            traceback.print_exc()

    def drop_numerical_values_in_categoricals(self, columns: list) -> None:
        try:
            for column in columns:
                if column in self.dataset.columns:
                    self.dataset = self.dataset[~self.dataset[column].apply(lambda x: isinstance(x, (int, float)))]
                    self.logger.info(f"Dropped numerical values in column: {column}")
                else:
                    self.logger.info(f"Column {column} not found in dataset.")
            self.logger.debug(f"Dataset after dropping numerical values in categoricals: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while dropping numerical values in categorical columns: {e}")
            traceback.print_exc()

    def drop_date_components(self):
        columns_to_drop = ['day', 'month', 'year', 'hour', 'minute', 'second']
        try:
            self.dataset.drop(columns=columns_to_drop, inplace=True)
            self.logger.info(f"Dropped columns: {', '.join(columns_to_drop)}")
            self.logger.debug(f"Dataset after dropping date components: {self.dataset.head()}")
        except KeyError as e:
            self.logger.error(f"One or more columns to drop not found in dataset: {e}")
        except Exception as e:
            self.logger.error(f"An error occurred while dropping columns: {e}")
            traceback.print_exc()

    def drop_specific_columns(self, columns_to_drop: list) -> None:
        try:
            self.dataset.drop(columns=columns_to_drop, inplace=True)
            self.logger.info(f"Dropped specific columns: {', '.join(columns_to_drop)}")
            self.logger.debug(f"Dataset after dropping specific columns: {self.dataset.head()}")
        except Exception as e:
            self.logger.error(f"An error occurred while dropping specific columns: {e}")
            traceback.print_exc()

# Usage example to load the dataset and run the pipeline
if __name__ == "__main__":
    de = DataEngineering()
    
    # Convert transactions_0.csv to transactions_0.parquet
    csv_filename = 'transactions_0.csv'
    parquet_filename = 'transactions_0.parquet'
    de.convert_csv_to_parquet(csv_filename, parquet_filename)
    
    # Append transactions_1.parquet to transactions_0.parquet and save as combined_transactions.parquet
    base_parquet = 'transactions_0.parquet'
    append_parquet = 'transactions_1.parquet'
    output_parquet = 'combined_transactions.parquet'
    de.append_parquet(base_parquet, append_parquet, output_parquet)
    
    # Load the combined Parquet dataset
    de.load_dataset(output_parquet)
    
    # Append data from transaction_2.json
    json_filename = 'transaction_2.json'
    de.append_json(json_filename)
    
    # Ensure columns are present and dataset not empty
    if de.dataset.empty:
        de.logger.info("The dataset is empty.")
    else:
        # Run the drop_date_components method
        de.drop_date_components()
        de.logger.info("Column names in dataset:", de.dataset.columns.tolist())

    # Data Cleaning Pipeline
    # Merge categories and get initial and merged counts
    initial_counts_list, merged_counts = de.merge_categories('category')

    # Print the initial category counts before merging
    de.logger.info("Initial category counts before merging:")
    for category, count in initial_counts_list:
        de.logger.info(f"{category}: {count}")

    # Print the merged category counts after normalization
    de.logger.info("\nMerged category counts after normalization:")
    for category, count in merged_counts.items():
        de.logger.info(f"{category}: {count}")

    de.drop_fraud_prefix()
    de.drop_columns(['sex', 'unix_time', 'job', 'dob'])
    de.logger.info(de.dataset.head())
    de.clean_missing_values()
    de.remove_duplicates()
    de.standardize_dates('trans_date_trans_time')
    de.trim_spaces()
    de.resolve_anomalous_dates('trans_date_trans_time')

    # Drop numerical values in categorical columns
    categorical_columns = ['merchant', 'category', 'city', 'state']
    de.drop_numerical_values_in_categoricals(categorical_columns)

    approved_categories = ['gas_transport', 'grocery_pos', 'home', 'shopping_pos', 'kids_pets', 'shopping_net', 'entertainment', 'food_dining', 'personal_care', 'health_fitness', 'misc_pos', 'misc_net', 'grocery_net', 'travel']
    de.categorical_data_validation('category', approved_categories)

    # Encode categorical features
    de.encode_categorical_features(categorical_columns)
    
    # Data Transformation
    de.expand_dates('trans_date_trans_time')
    de.categorize_transactions('amt')
    
    # Data Validation
    column_ranges = {'lat': (-90, 90), 'long': (-180, 180), 'amt': (0.01, 1000000000), 'merch_lat': (-90, 90), 'merch_long': (-180, 180), 'is_fraud': (0, 1)}
    essential_columns = ['category', 'amt']
    expected_dtypes = {'category': 'object', 'amt': 'float64', 'trans_date_trans_time': 'datetime64[ns]'}
    unique_columns = ['trans_num']

    # Perform range checks and clean dataset
    de.range_checks(column_ranges)
    de.null_checks(essential_columns)
    de.type_validation(expected_dtypes)
    de.uniqueness_validation(unique_columns)
    # Drop specific columns as the last operation
    columns_to_drop = [
        'merchant', 'first', 'last', 'street', 'lat', 'long',
        'merch_lat', 'merch_long', 'trans_date', 'trans_time', 'day', 'month', 'year', 'hour', 'minute', 'second',
    ]
    de.drop_specific_columns(columns_to_drop)

    # Save the processed dataset without adding blank rows
    try:
        # Convert object columns to string to avoid issues with fastparquet
        for col in de.dataset.select_dtypes(include=['object']).columns:
            de.dataset[col] = de.dataset[col].astype(str)
        
        print("Columns before saving:", de.dataset.columns.tolist())
        de.dataset.to_parquet('all_transactions.parquet', engine='fastparquet', index=False)
        de.logger.info("Processed dataset saved as all_transactions.parquet")
    except Exception as e:
        de.logger.error(f"An error occurred while saving the file: {e}")
        traceback.print_exc()
