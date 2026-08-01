# explainers/shap_explainer.py

import os
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")  # GUI backend for interactive display


def explain_with_shap(model, X, output_dir=None, metadata=None, class_idx=1):
    """
    Generate SHAP summary plot using original masking approach.
    Saves heatmap and raw feature importances.
    """
    try:
        print("\n🔍 Running SHAP explainability...")

        def mord_predict(X_):
            return model.predict(X_)

        # SHAP Masker
        masker = shap.maskers.Independent(X)

        # Explainer and SHAP values
        explainer = shap.Explainer(mord_predict, masker=masker)
        shap_values = explainer(X)

        # Handle multiclass (select one output)
        if isinstance(shap_values.values, list):
            shap_vals = shap.Explanation(
                values=shap_values.values[class_idx],
                base_values=shap_values.base_values[class_idx],
                data=shap_values.data,
                feature_names=shap_values.feature_names
            )
            shap_array = shap_values.values[class_idx]
        else:
            shap_vals = shap_values
            shap_array = shap_values.values

        # Plot heatmap-style summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X, plot_type="heatmap", show=False)
        plt.title(f"SHAP Feature Importance Heatmap (Class {class_idx})")
        plt.tight_layout()

        if output_dir:
            heatmap_path = os.path.join(output_dir, f"shap_heatmap_class{class_idx}.png")
            plt.savefig(heatmap_path)
            print(f"[✓] Saved SHAP heatmap to {heatmap_path}")

        plt.show(block=False)

        # Save mean SHAP values
        mean_abs = np.abs(shap_array)
        if mean_abs.ndim == 2:
            mean_abs = mean_abs.mean(axis=0)

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "MeanAbsSHAP": mean_abs
        }).sort_values(by="MeanAbsSHAP", ascending=False).set_index("Feature")

        if output_dir and metadata:
            json_data = {"metadata": metadata, "data": importance_df.to_dict()}
            json_path = os.path.join(output_dir, f"shap_feature_importance_class{class_idx}.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4, default=str)
            print(f"[✓] Saved SHAP values JSON to {json_path}")

    except Exception as e:
        print(f"[SHAP] Skipped due to error: {e}")
