import pandas as pd

class RuleBasedFiltering:
    """
    A rule-based recommender system that recommends top-rated movies overall or by genre.
    """

    def __init__(self, ratings_file, metadata_file):
        """
        Initialize the rule-based filter.

        :param ratings_file: Path to the ratings dataset (user, item, rating, timestamp).
        :param metadata_file: Path to the item metadata file (item, title, genres...).
        """
        # Load ratings and metadata
        self.ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
        self.metadata = pd.read_csv(
            metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                'unknown', 'Action', 'Adventure', 'Animation', "Children", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]
        )

        # Compute average rating per movie
        self.avg_ratings = self.ratings.groupby('item')['rating'].mean()

    def recommend_top(self, top_n=10):
        """
        Recommend the top_n movies overall by highest average rating.

        :param top_n: Number of movies to recommend.
        :return: DataFrame with columns ['item', 'title', 'avg_rating'] sorted descending.
        """
        # Identify top items by avg rating
        top_items = self.avg_ratings.sort_values(ascending=False).head(top_n).index
        df_top = self.metadata[self.metadata['item'].isin(top_items)].copy()
        df_top['avg_rating'] = df_top['item'].map(self.avg_ratings)
        return df_top[['item', 'title', 'avg_rating']].sort_values('avg_rating', ascending=False)

    def recommend_by_genre(self, genre, top_n=10):
        """
        Recommend the top_n movies within a specific genre by average rating.

        :param genre: Genre name matching a column in metadata (e.g., 'Comedy').
        :param top_n: Number of movies to recommend.
        :return: DataFrame with columns ['item', 'title', 'avg_rating'] sorted descending.
        """
        # Validate genre
        if genre not in self.metadata.columns:
            raise ValueError(f"Genre '{genre}' not found in metadata columns.")

        # Filter movies by genre
        genre_items = self.metadata[self.metadata[genre] == 1]['item']
        # Compute avg ratings for these items
        genre_avg = self.avg_ratings.loc[self.avg_ratings.index.isin(genre_items)]
        top_items = genre_avg.sort_values(ascending=False).head(top_n).index

        df_genre = self.metadata[self.metadata['item'].isin(top_items)].copy()
        df_genre['avg_rating'] = df_genre['item'].map(self.avg_ratings)
        return df_genre[['item', 'title', 'avg_rating']].sort_values('avg_rating', ascending=False)


if __name__ == "__main__":
    # Example usage
    rb = RuleBasedFiltering(
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )
    print("Top 10 Movies Overall:")
    print(rb.recommend_top(10))

    print("\nTop 10 Comedy Movies:")
    print(rb.recommend_by_genre('Comedy', 10))
