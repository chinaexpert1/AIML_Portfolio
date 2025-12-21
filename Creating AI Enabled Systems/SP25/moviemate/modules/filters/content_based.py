import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

class ContentBasedFiltering:
    """
    A content-based recommender system using TF-IDF and cosine similarity on genres.
    """

    def __init__(self, ratings_file, metadata_file):
        """
        Initialize the content-based filter.

        :param ratings_file: Path to the ratings dataset (user, item, rating, timestamp).
        :param metadata_file: Path to the item metadata file (with genre flags).
        """
        self.ratings_file = ratings_file
        self.metadata_file = metadata_file
        self.ratings = None
        self.items_metadata = None
        self.item_profiles = None
        self.similarity_matrix = None
        self.user_profiles = {}  # Cache for user profiles
        self._load_data()
        self._build_item_profiles()

    def _load_data(self):
        """Load ratings and metadata, and prepare genre features."""
        # Load user-item ratings
        self.ratings = pd.read_csv(
            self.ratings_file,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
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
        # Create a 'features' string by combining genres
        genre_cols = self.items_metadata.columns[6:]
        self.items_metadata['features'] = self.items_metadata[genre_cols] \
            .apply(lambda row: ' '.join(genre_cols[row == 1]), axis=1)

    def _build_item_profiles(self):
        """Vectorize genre features into TF-IDF item profiles and compute similarity matrix."""
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.items_metadata['features'])
        self.item_profiles = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=self.items_metadata['item']
        )
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    def _get_user_profile(self, user_id):
        """Aggregate item profiles weighted by user's ratings to form a user profile, with caching."""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        user_ratings = self.ratings[self.ratings['user'] == user_id]
        profile = np.zeros(self.item_profiles.shape[1])
        for _, row in user_ratings.iterrows():
            item = row['item']
            if item in self.item_profiles.index:
                profile += self.item_profiles.loc[item].values * row['rating']
        norm = np.linalg.norm(profile)
        user_profile = profile / norm if norm > 0 else profile
        self.user_profiles[user_id] = user_profile
        return user_profile

    def predict(self, user_id, item_id):
        """Compute similarity-based score between user profile and item profile."""
        if user_id not in self.user_profiles and user_id not in self.ratings['user'].unique():
            raise ValueError(f"User {user_id} not in ratings.")
        if item_id not in self.item_profiles.index:
            raise ValueError(f"Item {item_id} not in metadata.")
        user_profile = self._get_user_profile(user_id)
        item_vector = self.item_profiles.loc[item_id].values
        denom = np.linalg.norm(user_profile) * np.linalg.norm(item_vector)
        return np.dot(user_profile, item_vector) / denom if denom > 0 else 0.0

    def recommend(self, user_id, top_n=10):
        """Recommend top_n items for a user based on content similarity scores."""
        user_profile = self._get_user_profile(user_id)
        scores = self.item_profiles.dot(user_profile)
        # Exclude items already rated
        rated = set(self.ratings[self.ratings['user'] == user_id]['item'])
        scores = scores.drop(labels=rated, errors='ignore')
        top_items = scores.nlargest(top_n)
        df = self.items_metadata.set_index('item').loc[top_items.index]
        df = df[['title']].copy()
        df['score'] = top_items.values
        return df.reset_index()[['item', 'title', 'score']]

    def evaluate(self, sample_size=1000):
        """Evaluate RMSE between true ratings and predicted similarity scores scaled to rating range."""
        sample = self.ratings.sample(n=sample_size, random_state=42)
        y_true, y_pred = [], []
        for _, row in sample.iterrows():
            user, item, true_rating = row['user'], row['item'], row['rating']
            try:
                pred = self.predict(user, item) * 5
                y_true.append(true_rating)
                y_pred.append(pred)
            except ValueError:
                continue
        return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == '__main__':
    cb = ContentBasedFiltering(
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )
    print("Sample RMSE:", cb.evaluate(sample_size=100))
    print(cb.recommend(user_id=1, top_n=5))
