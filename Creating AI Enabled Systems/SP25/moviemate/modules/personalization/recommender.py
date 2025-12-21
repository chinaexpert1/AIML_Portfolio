import pandas as pd


class Recommender:
    """
    A hybrid recommender system that combines content-based and collaborative filtering.

    This class blends predictions from a content-based and a collaborative filtering model
    to generate ranked item recommendations for a given user.
    """

    def __init__(self, model, blending_weight=0.5):
        """
        Initialize the recommender system.

        Parameters:
        ----------
        model : object
            A model that supports `predict` and `items_metadata` for item scoring and metadata access.
        blending_weight : float, optional
            Weight for blending content-based and collaborative filtering predictions.
            Defaults to 0.5.
        """
        self.model = model
        self.blending_weight = blending_weight

    def rank_items(self, user_id, top_n=10):
        """
        Produce ranked item recommendations for a given user by blending predictions.

        Parameters:
        ----------
        user_id : int
            ID of the user for whom recommendations are generated.
        top_n : int, optional
            Number of top recommendations to return. Defaults to 10.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with ranked item IDs and their corresponding blended scores.
        """
        all_items = self.model.items_metadata['item'].unique()
        rankings = []

        for item_id in all_items:
            score = self.model.predict(user_id, item_id)
            rankings.append((item_id, score))

        ranked_items = sorted(rankings, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(ranked_items, columns=['item', 'score'])


if __name__ == "__main__":
    import sys
    sys.path.append("..")  # Adds higher directory to Python modules path.

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

    recommender = Recommender(model=model)
    rankings = recommender.rank_items(user_id=196, top_n=10)
    print(rankings)