#Feature Engineering
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


log_messages = []

def log(msg):
    print(msg)
    log_messages.append(str(msg))


class EngineerFeatures:

    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}

    def generate_features_from_json(self):
        """
        Generates engineered features from a single-row JSON payload.
        Uses the latest dataset from storage/datasets (which includes fraud_label)
        to compute aggregate thresholds and derive binary fraud features from the input.
        This includes:
        - Encoding categorical columns (using persisted LabelEncoders if available).
        - Calculating top fraud categories for 'category', 'job', 'hour', 'month_date',
            and 'day_of_week' based on the reference dataset.
        - Creating a suspicious amount flag based on quantile bins and fraud rates from the reference.
        """
        
        # --- Step 2: Log-transform the transaction amount ---
        if 'amt' in self.df.columns:
            self.df['log_amt'] = np.log1p(self.df['amt'])
        else:
            self.df['log_amt'] = 0

        # --- Step 3: Load the latest reference dataset and compute binary features ---
        import glob, os, pandas as pd
        dataset_folder = os.path.join("storage", "datasets")
        dataset_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
        if dataset_files:
            latest_dataset_file = max(dataset_files, key=os.path.getmtime)
            ref_df = pd.read_csv(latest_dataset_file)
            log(f"Loaded reference dataset: {latest_dataset_file}")
            
            # Helper: Get top fraud categories from ref_df
            def get_top_fraud_categories(ref_df, col, top_n=5, min_count=5):
                if col in ref_df.columns and 'fraud_label' in ref_df.columns:
                    fraud_counts = ref_df[ref_df['fraud_label'] == 1][col].value_counts()
                    fraud_counts = fraud_counts[fraud_counts >= min_count]
                    return fraud_counts.nlargest(top_n).index.tolist()
                else:
                    return []
            
            # Compute top lists based on the reference dataset:
            top_categories = get_top_fraud_categories(ref_df, 'category', top_n=6, min_count=20)
            top_jobs       = get_top_fraud_categories(ref_df, 'job', top_n=10, min_count=20)
            top_hours      = get_top_fraud_categories(ref_df, 'hour', top_n=6)
            top_months     = get_top_fraud_categories(ref_df, 'month_date', top_n=4)
            top_days       = get_top_fraud_categories(ref_df, 'day_of_week', top_n=3)
            # (Other fields such as 'merchant' or 'zip' are omitted since training used 7 features.)
            
            # For suspicious amount: compute quantile bins and fraud rates.
            if 'amt' in ref_df.columns and 'fraud_label' in ref_df.columns:
                ref_df['amt_bin'] = pd.qcut(ref_df['amt'], 20, duplicates='drop')
                amt_fraud_rates = ref_df.groupby('amt_bin').agg({'fraud_label': ['mean', 'count']})
                amt_fraud_rates.columns = ['fraud_rate', 'count']
                overall_fraud_rate = ref_df['fraud_label'].mean()
                significant_bins = amt_fraud_rates[amt_fraud_rates['fraud_rate'] > 2 * overall_fraud_rate].index.tolist()
                bin_intervals = pd.qcut(ref_df['amt'], q=20, duplicates='drop').cat.categories
                bin_edges = [interval.left for interval in bin_intervals] + [bin_intervals[-1].right]
                
                # Now for the single JSON row, determine its bin and suspicious flag.
                if 'amt' in self.df.columns:
                    value_amt = self.df.loc[0, 'amt']
                    bin_of_value = pd.cut(pd.Series([value_amt]), bins=bin_edges, include_lowest=True).iloc[0]
                    is_suspicious_amt = 1 if bin_of_value in significant_bins else 0
                else:
                    is_suspicious_amt = 0
            else:
                is_suspicious_amt = 0

            # Now compute binary features for categorical variables.
            # Because we have a single row, extract the value from row 0.
            if 'category' in self.df.columns:
                val_category = self.df.loc[0, 'category']
                is_top6_fraud_category = 1 if val_category in top_categories else 0
            else:
                is_top6_fraud_category = 0
            log("\n=== Top Categories with Fraud ===")
            log(top_categories)

            if 'job' in self.df.columns:
                val_job = self.df.loc[0, 'job']
                is_top10_fraud_job = 1 if val_job in top_jobs else 0
            else:
                is_top10_fraud_job = 0
            log("\n=== Top Jobs with Fraud ===")
            log(top_jobs)

            if 'hour' in self.df.columns:
                val_hour = self.df.loc[0, 'hour']
                is_top6_fraud_hour = 1 if val_hour in top_hours else 0
            else:
                is_top6_fraud_hour = 0
            log("\n=== Top Hours with Fraud ===")
            log(top_hours)

            if 'month_date' in self.df.columns:
                val_month = self.df.loc[0, 'month_date']
                is_top4_fraud_month = 1 if val_month in top_months else 0
            else:
                is_top4_fraud_month = 0
            log("\n=== Top Months with Fraud ===")
            log(top_months)

            if 'day_of_week' in self.df.columns:
                val_day = self.df.loc[0, 'day_of_week']
                is_top3_fraud_day = 1 if val_day in top_days else 0
            else:
                is_top3_fraud_day = 0
            log("\n=== Top Days with Fraud ===")
            log(top_days)

            # Insert computed binary features into the DataFrame.
            self.df['is_top6_fraud_category'] = is_top6_fraud_category
            self.df['is_top10_fraud_job'] = is_top10_fraud_job
            self.df['is_top6_fraud_hour'] = is_top6_fraud_hour
            self.df['is_top4_fraud_month'] = is_top4_fraud_month
            self.df['is_top3_fraud_day'] = is_top3_fraud_day
            self.df['is_suspicious_amt'] = is_suspicious_amt
        else:
            log("No reference dataset found in storage/datasets. Defaulting binary features to 0.")
            self.df['is_top6_fraud_category'] = 0
            self.df['is_top10_fraud_job'] = 0
            self.df['is_top6_fraud_hour'] = 0
            self.df['is_top4_fraud_month'] = 0
            self.df['is_top3_fraud_day'] = 0
            self.df['is_suspicious_amt'] = 0

        # --- Step 4: Retain only the desired features for prediction ---
        # The model was trained with exactly these 7 features.
        desired_features = [
            'is_top6_fraud_category', 
            'is_top10_fraud_job', 
            'is_top6_fraud_hour', 
            'is_top4_fraud_month',
            'is_top3_fraud_day', 
            'log_amt', 
            'is_suspicious_amt'
        ]
        self.df = self.df[desired_features]
        
        log("\n=== Engineered Features for JSON Input ===")
        log(self.df.head())
        return self.df, json.dumps(log_messages, indent=2)



        

    def get_top_fraud_categories(self, col, top_n=5, min_count=5):
        """
        Returns the top N categories in column `col` that are most associated with fraud.
        """
        fraud_counts = self.df[self.df['fraud_label'] == 1][col].value_counts()
        fraud_counts = fraud_counts[fraud_counts >= min_count]
        return fraud_counts.nlargest(top_n).index.tolist()


    def encode_columns(self):
        columns_to_encode = ['category', 'day_of_week', 'month_date']
        for col in ['category', 'day_of_week', 'month_date']:
            if col in self.df.columns:
                if col in self.label_encoders:
                    # Use the persisted transformer
                    self.df[col] = self.label_encoders[col].transform(self.df[col])
                    log(f"Transformed {col} using persisted encoder.")
                else:
                    # Fall back to fitting a new encoder if none is persisted.
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    self.label_encoders[col] = le
                    log(f"Encoded {col}: {len(le.classes_)} unique values")

        log("\nSample of encoded data:")
        log(self.df[columns_to_encode].head())

    def add_binary_fraud_features(self):
        top_merchants = self.get_top_fraud_categories('merchant', 5, 5)
        self.df['is_top5_fraud_merchant'] = self.df['merchant'].isin(top_merchants).astype(int)
        log("\n=== Top Merchants with Fraud ===")
        log(top_merchants)

        top_categories = self.get_top_fraud_categories('category', 6, 20)
        self.df['is_top6_fraud_category'] = self.df['category'].isin(top_categories).astype(int)
        log("\n=== Top Categories with Fraud ===")
        log(top_categories)

        top_jobs = self.get_top_fraud_categories('job', 10, 20)
        self.df['is_top10_fraud_job'] = self.df['job'].isin(top_jobs).astype(int)
        log("\n=== Top Jobs with Fraud ===")
        log(top_jobs)

        top_hours = self.get_top_fraud_categories('hour', 6)
        self.df['is_top6_fraud_hour'] = self.df['hour'].isin(top_hours).astype(int)
        log("\n=== Top Hours with Fraud ===")
        log(top_hours)

        top_months = self.get_top_fraud_categories('month_date', 4)
        self.df['is_top4_fraud_month'] = self.df['month_date'].isin(top_months).astype(int)
        log("\n=== Top Months with Fraud ===")
        log(top_months)

        top_days = self.get_top_fraud_categories('day_of_week', 3)
        self.df['is_top3_fraud_day'] = self.df['day_of_week'].isin(top_days).astype(int)
        log("\n=== Top Days with Fraud ===")
        log(top_days)

        top_zips = self.get_top_fraud_categories('zip', 2, 10)
        self.df['is_top2_fraud_zip'] = self.df['zip'].isin(top_zips).astype(int)
        log("\n=== Top ZIP Codes with Fraud ===")
        log(top_zips)


    def transform_amount_features(self):
        self.df['log_amt'] = np.log1p(self.df['amt'])
        self.df['amt_bin'] = pd.qcut(self.df['amt'], 20, duplicates='drop')
        amt_fraud_rates = self.df.groupby('amt_bin').agg({'fraud_label': ['mean', 'count']})
        amt_fraud_rates.columns = ['fraud_rate', 'count']

        # Injected block starts here
        overall_fraud_rate = self.df['fraud_label'].mean()
        significant_bins = amt_fraud_rates[amt_fraud_rates['fraud_rate'] > 2 * overall_fraud_rate].index.tolist()
        self.df['is_suspicious_amt'] = self.df['amt_bin'].isin(significant_bins).astype(int)

        log("\n=== Suspicious Transaction Amounts ===")
        log(f"Overall fraud rate: {overall_fraud_rate*100:.2f}%")
        log("Amount ranges with high fraud rates:")
        for bin_range in significant_bins:
            rate = amt_fraud_rates.loc[bin_range, 'fraud_rate'] * 100
            count = amt_fraud_rates.loc[bin_range, 'count']
            log(f"  {bin_range}: {rate:.2f}% fraud rate ({count} transactions)")

        self.df.drop('amt_bin', axis=1, inplace=True)
        # Injected block ends here

    def generate_features(self):
        self.encode_columns()
        self.add_binary_fraud_features()
        self.transform_amount_features()

        engineered_features = [
            'is_top6_fraud_category',
            'is_top10_fraud_job',
            'is_top6_fraud_hour',
            'is_top4_fraud_month',
            'is_top3_fraud_day',
            'log_amt',
            'is_suspicious_amt',
            'fraud_label'
        ]

        final_df = self.df[engineered_features]
        log("\n=== Engineered Features Summary ===")
        log(final_df.describe())
        log(f"\nFinal engineered dataset shape: {final_df.shape}")
        binary_count = sum(1 for col in final_df.columns if col.startswith(('is_', 'has_', 'unusual_')))
        log(f"Binary features created: {binary_count}")




        return final_df, json.dumps(log_messages, indent=2)

