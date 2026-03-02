import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.base import ClassifierMixin # For type hinting models


class EDAMetrics:
    @staticmethod
    def compute_confusion_elements(cm, class_idx):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - TP - FP - FN
        return TP, FN, FP, TN

    @staticmethod
    def compute_binary_metrics(TP, FN, FP, TN):
        # Precompute denominators (avoid repeated sums)
        pos_total = TP + FN
        neg_total = TN + FP
        pred_pos_total = TP + FP
        total = TP + TN + FP + FN

        # Core metrics
        recall = TP / pos_total if pos_total else 0  # Sensitivity / TPR
        specificity = TN / neg_total if neg_total else 0  # TNR
        precision = TP / pred_pos_total if pred_pos_total else 0
        accuracy = (TP + TN) / total if total else 0

        # Derived metrics
        fpr = 1 - specificity  # Faster than recomputing
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        youden = recall + specificity - 1
        plr = (recall / fpr) if fpr else float("inf")

        return {
            "sensitivity": recall,
            "specificity": specificity,
            "youden_index": youden,
            "accuracy" : accuracy
        }

    @classmethod
    def binary_classification_metrics(cls, cm, labels="", verbosity=1):
        tn, fp, fn, tp = cm.ravel()
        metrics = cls.compute_binary_metrics(tp, fn, fp, tn)

        if verbosity > 0:
            for k in ['sensitivity', 'specificity', 'youden_index', 'accuracy']:
                print(f"{k.replace('_', ' ')}: {metrics[k]:.3f}")
            print("")
        if verbosity > 1:
            print(f"tpr: {metrics['tpr']:.3f}")
            print(f"plr: {metrics['plr']:.3f}")
            print("---")
            if labels:
                print(labels)
            print(f"TN {tn}   FP {fp}")
            print(f"FN {fn}   TP {tp}")
            print(cm)
            print("")

        return metrics

    @classmethod
    def multiclass_metrics(cls, cm, labels="", result_aggregation='by_statistics', verbosity=1):
        n_classes = cm.shape[0]
        labels = labels or list(range(n_classes))

        per_class_metrics = []
        supports = []

        for i in range(n_classes):
            TP, FN, FP, TN = cls.compute_confusion_elements(cm, i)
            metrics = cls.compute_binary_metrics(TP, FN, FP, TN)
            support = np.sum(cm[i, :])
            metrics['support'] = support
            per_class_metrics.append(metrics)
            supports.append(support)

            if verbosity > 1:
                print(f"Class {labels[i]}:")
                print(f"  TP = {TP}, FN = {FN}, FP = {FP}, TN = {TN}, Support = {support}")
                for key in ['sensitivity', 'specificity', 'youden_index']:
                    print(f"  {key.capitalize().replace('_', ' ')} = {metrics[key]:.3f}")
                print("")

        correct = np.trace(cm)
        total = np.sum(cm)
        accuracy = correct / total

        support_arr = np.array(supports)
        weights = support_arr / np.sum(support_arr)

        macro_avg = {
            k: np.mean([m[k] for m in per_class_metrics])
            for k in ['sensitivity', 'specificity', 'youden_index']
        }
        weighted_avg = {
            k: np.average([m[k] for m in per_class_metrics], weights=weights)
            for k in ['sensitivity', 'specificity', 'youden_index']
        }

        if verbosity > 0:
            print("Macro-Averaged Metrics:")
            for k, v in macro_avg.items():
                print(f"  {k.capitalize()} (Macro) = {v:.3f}")
            print("\nWeighted-Averaged Metrics:")
            for k, v in weighted_avg.items():
                print(f"  {k.capitalize()} (Weighted) = {v:.3f}")
            print("")

        if result_aggregation == 'by_statistics':
            return {
                "per_class": {
                    key: [m[key] for m in per_class_metrics]
                    for key in ['sensitivity', 'specificity', 'youden_index', 'support']
                },
                "macro_avg": {
                    **macro_avg,
                },
                "weighted_avg": weighted_avg
            }
        else:  # 'by_class'
            return {
                labels[i]: {
                    "support": per_class_metrics[i]["support"],
                    "sensitivity": [
                        per_class_metrics[i]["sensitivity"],
                        macro_avg["sensitivity"],
                        weighted_avg["sensitivity"]
                    ],
                    "specificity": [
                        per_class_metrics[i]["specificity"],
                        macro_avg["specificity"],
                        weighted_avg["specificity"]
                    ],
                    "youden_index": [
                        per_class_metrics[i]["youden_index"],
                        macro_avg["youden_index"],
                        weighted_avg["youden_index"]
                    ]
                }
                for i in range(n_classes)
            }

    def plot_roc_binary_from_model(model: ClassifierMixin, X_val: pd.DataFrame, y_val: pd.Series, class_label: str = "Positive Class"):
        """
        Generates and plots the ROC curve for binary classification directly from the model.

        Args:
            model (ClassifierMixin): Trained scikit-learn compatible binary classifier.
            X_val (pd.DataFrame): Validation/test features.
            y_val (pd.Series): True binary labels for validation/test set.
            class_label (str): Label for the positive class to display in the plot title/legend.
        """
        if not hasattr(model, 'predict_proba'):
            print(f"Error: Model {type(model).__name__} does not have a predict_proba method. Cannot plot ROC curve.")
            return

        y_proba = model.predict_proba(X_val)[:, 1] # Get probabilities for the positive class
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) for {class_label}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_roc_multiclass_from_model(model: ClassifierMixin, X_val: pd.DataFrame, y_val: pd.Series, class_names: list):
        """
        Generates and plots ROC curves for multiclass classification directly from the model:
        1. All curves (micro-average, macro-average, and individual classes) on one graph.
        2. Individual ROC curves for each class on separate subplots.

        Args:
            model (ClassifierMixin): Trained scikit-learn compatible multiclass classifier.
            X_val (pd.DataFrame): Validation/test features.
            y_val (pd.Series): True labels for validation/test set.
            class_names (list): List of names for each class (e.g., ['Negative', 'Possible', 'Probable', 'Confirmed']).
        """
        if not hasattr(model, 'predict_proba'):
            print(f"Error: Model {type(model).__name__} does not have a predict_proba method. Cannot plot ROC curve.")
            return

        y_proba = model.predict_proba(X_val)
        n_classes = len(class_names)

        # Binarize the true labels for one-vs-rest ROC
        # Ensure classes are correctly ordered for label_binarize
        unique_classes_in_y_val = np.unique(y_val)
        y_true_binarized = label_binarize(y_val, classes=np.sort(unique_classes_in_y_val))

        # Handle cases where model might predict fewer classes than expected by class_names
        # This can happen if some classes are missing in y_val or model's predictions
        if y_proba.shape[1] != n_classes:
            print(f"Warning: Model predicts {y_proba.shape[1]} classes, but {n_classes} class_names provided.")
            print("Adjusting y_proba to match class_names by adding zero columns for missing classes if necessary.")
            # Create a full probability array with zeros for missing classes
            full_y_proba = np.zeros((y_proba.shape[0], n_classes))
            model_classes = model.classes_ # Get classes known to the model

            for i, class_name in enumerate(class_names):
                if class_name in model_classes:
                    # Find the index of this class in the model's output
                    model_class_idx = np.where(model_classes == class_name)[0][0]
                    full_y_proba[:, i] = y_proba[:, model_class_idx]
                # else: it remains zero, correctly representing no probability for that class
            y_proba = full_y_proba


        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # --- Plot 1: All ROC curves on one graph ---
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-average ROC (area = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) for Multiclass')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True)
        plt.show()

        # --- Plot 2-5: Individual ROC curves for each class ---
        # Determine grid size for subplots
        rows = int(np.ceil(n_classes / 2)) # 2 columns per row
        fig, axes = plt.subplots(rows, 2, figsize=(14, 6 * rows)) # Adjust layout dynamically
        axes = axes.flatten()

        # Reset colors for individual plots
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])

        for i, ax in enumerate(axes):
            if i < n_classes:
                ax.plot(fpr[i], tpr[i], color=next(colors), lw=2,
                        label=f'ROC curve (area = {roc_auc[i]:.2f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC for Class: {class_names[i]}')
                ax.legend(loc="lower right")
                ax.grid(True)
            else:
                fig.delaxes(ax) # Remove empty subplots

        plt.tight_layout()
        plt.show()

    def display_tabular_stats(df: pd.DataFrame, include_object: bool = False):
        """
        Displays basic tabular statistics and info for a DataFrame.
        """
        print("\n--- DataFrame Info ---")
        df.info()

        print("\n--- Descriptive Statistics (Numerical Columns) ---")
        print(df.describe())

        if include_object:
            print("\n--- Descriptive Statistics (Categorical/Object Columns) ---")
            print(df.select_dtypes(include=['object', 'category']).describe())

        print("\n--- Missing Values Count ---")
        print(df.isnull().sum().sort_values(ascending=False))
        print("\n--- Unique Values Count for each column ---")
        print(df.nunique().sort_values(ascending=False))

    def plot_missing_values(df: pd.DataFrame, figsize: tuple = (10, 6)):
        """
        Plots a bar chart of the count of missing values per column.
        """
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        if missing_counts.empty:
            print("No missing values to plot.")
            return

        plt.figure(figsize=figsize)
        sns.barplot(x=missing_counts.values, y=missing_counts.index, palette='viridis')
        plt.title("Count of Missing Values per Column")
        plt.xlabel("Number of Missing Values")
        plt.ylabel("Column Name")
        plt.show()

    def plot_unique_values_count(df: pd.DataFrame, figsize: tuple = (10, 6)):
        """
        Plots a bar chart of the count of unique values per column.
        """
        unique_counts = df.nunique().sort_values(ascending=False)

        plt.figure(figsize=figsize)
        sns.barplot(x=unique_counts.values, y=unique_counts.index, palette='plasma')
        plt.title("Count of Unique Values per Column")
        plt.xlabel("Number of Unique Values")
        plt.ylabel("Column Name")
        plt.show()

    def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list, bins: int = 30, figsize: tuple = (12, 5)):
        """
        Plots histograms and box plots for numerical columns.
        """
        print("\n--- Plotting Numerical Feature Distributions ---")
        for col in numerical_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                fig, axes = plt.subplots(1, 2, figsize=figsize)

                # Histogram
                sns.histplot(df[col], kde=True, bins=bins, ax=axes[0])
                axes[0].set_title(f'Distribution of {col}')

                # Box Plot
                sns.boxplot(x=df[col], ax=axes[1])
                axes[1].set_title(f'Box Plot of {col}')

                plt.tight_layout()
                plt.show()
            else:
                print(f"Warning: Column '{col}' not found or is not numerical in DataFrame. Skipping plot.")

    def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list, figsize: tuple = (8, 4)):
        """
        Plots bar plots for categorical columns.
        """
        print("\n--- Plotting Categorical Feature Distributions ---")
        for col in categorical_cols:
            if col in df.columns:
                plt.figure(figsize=figsize)
                sns.countplot(x=df[col], palette='viridis', order=df[col].value_counts().index)
                plt.title(f'Count Plot of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            else:
                print(f"Warning: Categorical column '{col}' not found in DataFrame. Skipping plot.")

    def plot_heatmap(corr_matrix, figsize=(6, 4)):
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()

    def plot_heatmap_thresholded(corr_matrix, threshold=0.9, figsize=(6, 4)):
        mask = corr_matrix.abs() < threshold
        mask |= np.eye(len(corr_matrix)).astype(bool)  # keep diagonal clear

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix (Only > 0.8 Shown)")
        plt.show()

    def find_highly_correlated_features(X, y, threshold=0.8):
        """
        Finds pairs of highly correlated features in X and reports their correlation
        to the target variable y.

        Args:
            X (pd.DataFrame): DataFrame of features.
            y (pd.Series): Series of the target variable.
            threshold (float): The correlation threshold to consider as 'high'.

        Returns:
            pd.DataFrame: A DataFrame with details of highly correlated feature pairs.
        """
        # 1. Combine features and target into a single DataFrame
        full_df = pd.concat([X, y], axis=1)

        # 2. Calculate the correlation matrix
        corr_matrix = full_df.corr().abs()

        # 3. Get the names of all feature columns (excluding y)
        feature_cols = X.columns

        # 4. Find highly correlated pairs
        correlated_pairs = []

        # We iterate through the upper triangle of the correlation matrix to avoid duplicates
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                feature1 = feature_cols[i]
                feature2 = feature_cols[j]
                corr_val = corr_matrix.loc[feature1, feature2]

                if corr_val > threshold:
                    corr_to_y1 = corr_matrix.loc[feature1, y.name]
                    corr_to_y2 = corr_matrix.loc[feature2, y.name]

                    correlated_pairs.append({
                        'Feature 1': feature1,
                        'Feature 2': feature2,
                        'Pair Correlation': corr_val,
                        f'Correlation with {y.name} (1)': corr_to_y1,
                        f'Correlation with {y.name} (2)': corr_to_y2
                    })

        return pd.DataFrame(correlated_pairs)

    def get_features_to_drop(X, y, threshold=0.8):
        correlated_pairs = EDAMetrics.find_highly_correlated_features(X, y, threshold)
        features_to_drop = set()

        for _, row in correlated_pairs.iterrows():
            f1, f2 = row['Feature 1'], row['Feature 2']
            corr_y1 = row[f'Correlation with {y.name} (1)']
            corr_y2 = row[f'Correlation with {y.name} (2)']

            # Drop the feature with the weaker correlation to the target
            if corr_y1 < corr_y2:
                features_to_drop.add(f1)
            else:
                features_to_drop.add(f2)

        return list(features_to_drop)