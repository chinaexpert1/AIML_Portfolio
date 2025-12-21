from scipy.stats import ks_2samp
import numpy as np


class ContinuousLearner:
    """
    A class to evaluate whether a machine learning model needs retraining based on statistical tests
    comparing baseline and production performance metrics (e.g., RMSE).
    """

    def __init__(self, baseline_rmse):
        """
        Initialize the ContinuousLearner.

        Args:
            baseline_rmse (array-like): RMSE values for the baseline dataset.
        """
        self.baseline_rmse = baseline_rmse

    def detect_drift(self, production_rmse, return_pvalue=False):
        """
        Determine if the model needs retraining based on a KS test comparing
        baseline and production RMSE distributions.

        Args:
            production_rmse (array-like): RMSE values for the production dataset.
            return_pvalue (bool): Whether to return the p-value along with the drift result.

        Returns:
            bool or tuple: 
                - If `return_pvalue` is False: A boolean indicating whether retraining is needed.
                - If `return_pvalue` is True: A tuple containing the boolean and the p-value.
        """
        if self.baseline_rmse is None:
            raise ValueError("Baseline RMSE has not been provided. Initialize with baseline RMSE values.")

        # Perform KS test
        stat, p_value = ks_2samp(
            self.baseline_rmse,
            production_rmse,
            alternative="greater"
        )

        # Determine if retraining is needed
        if return_pvalue:
            return p_value < 0.05, p_value
        else:
            return p_value < 0.05


if __name__ == "__main__":
    # Simulated data
    baseline_rmse = np.random.normal(0.0, 0.5, 100)
    production_rmse = np.random.normal(0.1, 0.5, 2000)

    learner = ContinuousLearner(baseline_rmse=baseline_rmse)
    retrain = learner.detect_drift(production_rmse)
    print(retrain)