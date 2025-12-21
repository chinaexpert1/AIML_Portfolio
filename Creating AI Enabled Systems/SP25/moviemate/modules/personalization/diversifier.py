import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


# Helper Functions
def calculate_intra_list_diversity(selected_indices, genre_vectors):
    """
    Calculate the Intra-List Diversity (ILD).

    Args:
        selected_indices (list): Indices of selected items.
        genre_vectors (array-like): One-hot encoded genre vectors.

    Returns:
        float: Average dissimilarity among selected items.
    """
    if len(selected_indices) < 2:
        return 1
    selected_vectors = genre_vectors[selected_indices]
    similarities = cosine_similarity(selected_vectors)
    dissimilarities = 1 - similarities
    return np.mean(dissimilarities[np.triu_indices_from(dissimilarities, k=1)])


def calculate_shannon_entropy(selected_indices, genre_vectors):
    """
    Calculate Shannon Entropy based on genre distributions.

    Args:
        selected_indices (list): Indices of selected items.
        genre_vectors (array-like): One-hot encoded genre vectors.

    Returns:
        float: Shannon entropy score.
    """
    if not selected_indices:
        return 0
    genre_counts = genre_vectors[selected_indices].sum(axis=0)
    genre_distribution = genre_counts / genre_counts.sum()
    return entropy(genre_distribution)


def calculate_novelty(selected_indices, popularity_dict):
    """
    Calculate novelty based on item popularity.

    Args:
        selected_indices (list): Indices of selected items.
        popularity_dict (dict): Dictionary mapping items to their popularity.

    Returns:
        float: Average novelty score.
    """
    if not selected_indices:
        return 0
    novelty_scores = [-np.log(popularity_dict.get(item, 1e-10)) for item in selected_indices]
    return np.mean(novelty_scores)


def calculate_pairwise_dissimilarity(selected_indices, feature_vectors):
    """
    Calculate pairwise dissimilarity based on feature embeddings.

    Args:
        selected_indices (list): Indices of selected items.
        feature_vectors (array-like): Embedding vectors for items.

    Returns:
        float: Average dissimilarity among selected items.
    """
    if len(selected_indices) < 2:
        return 1
    selected_vectors = feature_vectors[selected_indices]
    similarities = cosine_similarity(selected_vectors)
    dissimilarities = 1 - similarities
    return np.mean(dissimilarities[np.triu_indices_from(dissimilarities, k=1)])


def calculate_gini_index(selected_indices):
    """
    Calculate Gini Index for item distribution.

    Args:
        selected_indices (list): Indices of selected items.

    Returns:
        float: Gini index score.
    """
    counts = pd.Series(selected_indices).value_counts().values
    n = len(counts)
    if n == 0:
        return 0
    sorted_counts = np.sort(counts)
    cumulative_sum = np.cumsum(sorted_counts)
    return 1 - (2 / (n - 1)) * np.sum((n - np.arange(n)) * sorted_counts / cumulative_sum[-1])


def calculate_rank_diversity(selected_indices, genre_vectors):
    """
    Calculate rank-based diversity weighted by rank.

    Args:
        selected_indices (list): Indices of selected items.
        genre_vectors (array-like): One-hot encoded genre vectors.

    Returns:
        float: Rank-based diversity score.
    """
    diversity_scores = []
    for i, rank in enumerate(selected_indices):
        for j in range(i + 1, len(selected_indices)):
            sim = cosine_similarity(
                genre_vectors[rank].reshape(1, -1),
                genre_vectors[selected_indices[j]].reshape(1, -1)
            )
            diversity_scores.append(1 - sim[0, 0])
    return np.mean(diversity_scores) if diversity_scores else 1


class Diversifier:
    """
    A class to rerank recommendations by balancing relevance and diversity.
    """

    def __init__(self, diversity_measures=None, top_n=None, metadata_file="storage/u.item"):
        """
        Initialize the Diversifier.

        Args:
            diversity_measures (list): List of functions for diversity measures.
            top_n (int): Number of recommendations to rerank.
            metadata_file (str): Path to the metadata file.
        """
        self.diversity_measures = diversity_measures or []
        self.movies = pd.read_csv(
            metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'movieId', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )
        self.top_n = top_n

    def rerank(self, initial_recommendations, alpha=0.7):
        """
        Perform reranking to balance relevance and diversity.

        Args:
            initial_recommendations (DataFrame): DataFrame with movieId and relevance_score.
            alpha (float): Weight for relevance vs diversity.

        Returns:
            DataFrame: Reranked recommendations with scores.
        """
        selected_movies = []
        reranked = []
        genre_vector = self.movies.iloc[:, 5:].values

        for _, row in initial_recommendations.iterrows():
            movie_id = int(row['item'])
            relevance_score = row['score']
            movie_idx = int(self.movies[self.movies['movieId'] == movie_id].index[0])

            # Calculate diversity score
            diversity_score = 0
            for measure in self.diversity_measures:
                diversity_score += measure(selected_movies, genre_vector)

            # Normalize diversity score
            diversity_score = diversity_score * 5.0 / len(self.diversity_measures) if self.diversity_measures else 0

            # Combine relevance and diversity scores
            final_score = alpha * relevance_score + (1 - alpha) * diversity_score
            reranked.append((movie_id, relevance_score, diversity_score, final_score))

            # Update selected movies
            selected_movies.append(movie_idx)

        # Sort by final score
        reranked = sorted(reranked, key=lambda x: x[3], reverse=True)
        return pd.DataFrame(reranked, columns=['item', 'relevance_score', 'diversity_score', 'total_score']).head(self.top_n)


if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from moviemate.modules.adaptive.filters.collaborative import CollaborativeFiltering
    from surprise import SVD

    svd_params = {
        'n_factors': 200,
        'n_epochs': 100,
        'lr_all': 0.01,
        'reg_all': 0.1
    }
    model = CollaborativeFiltering(
        algorithm=SVD(**svd_params),
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )
    model.fit()

    from recommender import Recommender

    recommender = Recommender(model=model)
    initial_rankings = recommender.rank_items(user_id=196, top_n=10)
    print("Initial Rankings")
    print(initial_rankings)

    # Define diversity functions
    diversity_measures = [
        lambda selected, genres: calculate_intra_list_diversity(selected, genres),
    ]

    # Initialize reranker with diversity functions
    reranker = Diversifier(diversity_measures=diversity_measures, top_n=10)

    # Rerank recommendations
    reranked = reranker.rerank(initial_rankings, alpha=0.9)
    print()
    print("Reranked Recommendations:")
    print(reranked)