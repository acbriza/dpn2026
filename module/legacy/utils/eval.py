import gower
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class DiceEvaluator:

    def __init__(self, dice_explainer, data):
        self.explainer = dice_explainer
        self.feature_mins = data.min()
        self.feature_maxs = data.max()

    # ---------------------------
    # 1. Proximity
    # ---------------------------
    def _gower_distance(self, x, y):
        diffs = []
        for col in x.index:
            if pd.api.types.is_numeric_dtype(x[col]):
                R = self.feature_maxs[col] - self.feature_mins[col] + 1e-6
                diffs.append(abs(x[col] - y[col]) / R)
            else:
                diffs.append(0 if x[col] == y[col] else 1)
        return np.mean(diffs)

    def _get_proximity(self, instance, cfs):
        inst = instance.iloc[0]  # assume one row
        distances = [self._gower_distance(inst, cf) for _, cf in cfs.iterrows()]
        return float(np.mean(distances))

    # ---------------------------
    # 2. Sparsity
    # ---------------------------
    @staticmethod
    def _sparsity(instance, cfs):
        """
        Sparsity: average number of features changed (as per literature definition).
        """
        inst = instance.values.flatten()
        m = len(cfs)            # number of counterfactuals
        rho = len(inst)         # number of features
        total_changes = 0

        for _, cf in cfs.iterrows():
            # Count number of features where instance != counterfactual
            total_changes += np.sum(cf.values != inst)

        # Apply formula: (1 / (m * rho)) * total_changes
        return total_changes / (m * rho)

    # ---------------------------
    # 3. Diversity
    # ---------------------------
    @staticmethod
    def _calculate_diversity_det(cfs):
        """
        Measures diversity using the determinant of the Gower similarity matrix.

        Args:
            cfs (pd.DataFrame): The DataFrame of counterfactuals, with features only.

        Returns:
            float: The determinant of the similarity matrix.
        """
        if len(cfs) < 2:
            return 0.0

        # Calculate the Gower similarity matrix
        similarity_matrix = 1 - gower.gower_matrix(cfs)

        # Calculate the determinant of the matrix
        det = np.linalg.det(similarity_matrix)

        return float(det)

    @staticmethod
    def _calculate_diversity_lcc(self, cfs: pd.DataFrame, num_total_labels: int, label_column: str):
        """
        Calculates diversity using the determinantal approach, weighted by the
        Local Coverage Coefficient (LCC).

        Args:
            cfs (pd.DataFrame): The DataFrame of counterfactuals, including the label column.
            num_total_labels (int): The total number of unique class labels in your dataset.
            label_column (str): The name of the label column in the cfs DataFrame.

        Returns:
            float: The weighted diversity score.
        """
        if len(cfs) < 2:
            return 0.0

        # 1. Calculate the determinantal diversity
        # Drop the label column before passing to the determinantal function
        diversity_score = self._calculate_diversity_det(cfs.drop(columns=[label_column]))

        # 2. Get the number of unique labels among the generated counterfactuals
        unique_cf_labels = len(cfs[label_column].unique())

        # 3. Calculate the local coverage coefficient (LCC)
        # The (num_total_labels - 1) is from the paper's Equation 10
        if (num_total_labels - 1) == 0:
            lcc = 0
        else:
            lcc = unique_cf_labels / (num_total_labels - 1)

        # 4. Calculate the final diversity_lcc score (Equation 11)
        diversity_lcc_score = lcc * diversity_score

        return diversity_lcc_score

    # ---------------------------
    # 4. Plausibility
    # ---------------------------
    @staticmethod
    def _plausibility(instance, cfs):
        """
        Checks whether CFs lie in the data distribution.
        Here: z-score outlier detection (low plausibility if far from mean).
        """
        plaus_scores = []
        for _, cf in cfs.iterrows():
            # simple heuristic: smaller deviation = more plausible
            zscores = (cf - cfs.mean()) / (cfs.std() + 1e-6)
            plaus_scores.append(np.exp(-np.mean(np.abs(zscores))))
        return float(np.mean(plaus_scores))

    # ---------------------------
    # 5. Feasibility
    # ---------------------------
    # @staticmethod
    # def _feasibility(cfs: pd.DataFrame, data: pd.DataFrame, label_col: str, k: int):
    #     """
    #     Calculates the average distance to k-nearest neighbors for a set of counterfactuals.
    #
    #     Args:
    #         cfs (pd.DataFrame): The DataFrame of counterfactuals.
    #         data (pd.DataFrame): The full dataset used for training, including features and labels.
    #         label_col (str): The name of the label column.
    #         k (int): The number of nearest neighbors to consider.
    #
    #     Returns:
    #         float: The average feasibility score. A smaller value indicates higher feasibility.
    #     """
    #     feasibility_scores = []
    #
    #     # Get features from the full dataset
    #     data_X = data.drop(columns=[label_col])
    #
    #     # Fit a NearestNeighbors model on the full dataset
    #     nn_model = NearestNeighbors(n_neighbors=k)
    #     nn_model.fit(data_X)
    #
    #     for _, cf_row in cfs.iterrows():
    #         # Find the k-nearest neighbors and their distances to the counterfactual
    #         distances, _ = nn_model.kneighbors(cf_row.drop(label_col).values.reshape(1, -1))
    #
    #         # Calculate the average distance for this counterfactual
    #         avg_dist = np.mean(distances)
    #         feasibility_scores.append(avg_dist)
    #
    #     return float(np.mean(feasibility_scores))

    def _feasibility(instance, cfs):

        """

        Check if CFs respect feature constraints (e.g., non-negative, categorical valid values).

        For now: penalize negative values (assumes non-negativity constraint).

        """

        feas_scores = []

        for _, cf in cfs.iterrows():

            valid = np.all(cf.values >= 0) # simple feasibility check

        feas_scores.append(1.0 if valid else 0.0)

        return float(np.mean(feas_scores))

    # ---------------------------
    # 6. Robustness
    # ---------------------------
    @staticmethod
    def _robustness(instance, cfs):
        """
        Stability of CFs under small perturbations of the instance.
        Here: re-apply small noise to instance and check CF distance.
        """
        inst = instance.values.flatten()
        robustness_scores = []
        noise = np.random.normal(0, 0.01, size=inst.shape)
        perturbed_inst = inst + noise
        for _, cf in cfs.iterrows():
            dist = np.linalg.norm(cf.values - perturbed_inst, ord=2)
            robustness_scores.append(1 / (1 + dist))  # smaller distance = more robust
        return float(np.mean(robustness_scores))

    # ---------------------------
    # Aggregation
    # ---------------------------
    def get_stats(self, data, classification_type, verbosity=1):
        examples_stats = []
        for example in self.explainer.cf_examples_list:

            test_instance = example.test_instance_df
            cfs = example.final_cfs_df_sparse

            assert cfs is not None

            if classification_type == 'binary':
                label_column = 'Confirmed_Binary_DPN'
            elif classification_type == 'multiclass':
                label_column = 'DPN_Status'

            stats = {
                'test_instance': test_instance,
                'cfs': cfs,
                'proximity': self._get_proximity(test_instance, cfs),
                'sparsity': self._sparsity(test_instance, cfs),
                'diversity': self._calculate_diversity_det(cfs),
                'diversity_lcc': self._calculate_diversity_lcc(self, cfs=cfs, num_total_labels=1, label_column=label_column),
                'plausibility': self._plausibility(test_instance, cfs),
                'feasibility': self._feasibility(test_instance, cfs),
                'robustness': self._robustness(test_instance, cfs)
            }

            examples_stats.append(stats)

        average_stats = {
            'proximity': np.mean([s['proximity'] for s in examples_stats]),
            'sparsity': np.mean([s['sparsity'] for s in examples_stats]),
            'diversity': np.mean([s['diversity'] for s in examples_stats]),
            'diversity_lcc': np.mean([s['diversity_lcc'] for s in examples_stats]),
            'plausibility': np.mean([s['plausibility'] for s in examples_stats]),
            'feasibility': np.mean([s['feasibility'] for s in examples_stats]),
            'robustness': np.mean([s['robustness'] for s in examples_stats])
        }

        if verbosity >= 1:
            return {
                'examples_stats': examples_stats,
                'average_stats': average_stats
            }

        else:
            return average_stats
