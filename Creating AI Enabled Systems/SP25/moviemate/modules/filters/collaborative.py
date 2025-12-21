import pandas as pd
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import Reader

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class CollaborativeFiltering:
    """
    A collaborative filtering recommender system using Surprise matrix factorization.

    This class predicts user-item ratings leveraging algorithms like SVD.
    """

    def __init__(self, ratings_file, metadata_file, algorithm=None, test_size=0.2, random_state=42):
        """
        Initialize the recommender system and load data.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file (user, item, rating, timestamp).
        metadata_file : str
            Path to the item metadata file (with genres and titles).
        algorithm : object, optional
            A Surprise algorithm instance (e.g., SVD()). If None, defaults to SVD().
        test_size : float, optional
            Proportion of the dataset used for validation. Defaults to 0.2.
        random_state : int, optional
            Random seed for reproducibility. Defaults to 42.
        """
        # Default algorithm
        self.algorithm = algorithm if algorithm is not None else SVD()
        self.test_size = test_size
        self.random_state = random_state
        self.trainset = None
        self.validset = None
        self.model = None
        self.data_file = ratings_file
        self.metadata_file = metadata_file
        self.data = None
        self.items_metadata = None
        self._load_data()

    def _load_data(self):
        """Load rating data and item metadata, then split into train/validation sets."""
        # Load ratings
        df = pd.read_csv(
            self.data_file,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

        # Load item metadata
        self.items_metadata = pd.read_csv(
            self.metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )

        # Split into training and validation
        self.trainset, self.validset = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def fit(self):
        """Train the collaborative filtering model on the training set."""
        if self.trainset is None:
            raise ValueError("Training set is not available. Ensure `_load_data` has been called.")
        self.model = self.algorithm
        self.model.fit(self.trainset)

    def evaluate(self):
        """Evaluate the model by computing RMSE on the validation set."""
        if self.validset is None:
            raise ValueError("Validation set is not available. Cannot evaluate.")
        predictions = self.model.test(self.validset)
        return accuracy.rmse(predictions)

    def predict(self, user_id, item_id):
        """Predict the rating a user would give to an item."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call `fit()` first.")
        return self.model.predict(user_id, item_id).est

    def rank_items(self, user_id, top_n=10):
        """
        Generate top-N item recommendations for a user by predicting all unknown ratings.

        Returns a DataFrame with columns ['item', 'score'] sorted descending.
        """
        if self.items_metadata is None:
            raise ValueError("Item metadata not loaded.")
        # Predict score for each item
        scores = []
        for item in self.items_metadata['item']:
            try:
                score = self.predict(user_id, item)
                scores.append((item, score))
            except Exception:
                continue
        # Select top_n
        top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top_scores, columns=['item', 'score'])

if __name__ == "__main__":
    # Example usage
    cf = CollaborativeFiltering(
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )
    cf.fit()
    print("Validation RMSE:", cf.evaluate())
    print("Top-10 for user 196:", cf.rank_items(196, 10))
