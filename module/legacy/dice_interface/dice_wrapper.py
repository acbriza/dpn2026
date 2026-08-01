# dice_interface/dice_wrapper.py

import dice_ml
import pandas as pd


class DiCEWrapper:
    def __init__(self, dice_model, dice_data):
        """
        Initialize the DiCE explainer with the given DiCE model and data.
        """
        self.cls_batches = {}
        self.dice = dice_ml.Dice(
            dice_data,
            dice_model,
            method="genetic"
        )

        self.dice_data = dice_data
        self.dice_model = dice_model

    def generate_and_show(self, query_instances, features_to_vary, desired_class, total_CFs=5):
        """
        Generate counterfactuals for a single input or DataFrame.

        :param query_instances: DataFrame row(s) or dict of input features.
        :param features_to_vary: list of feature names to vary.
        :param desired_class: target class for the counterfactual.
        :param total_CFs: number of counterfactuals to generate.
        """
        # If it's a single row Series, convert to DataFrame
        if hasattr(query_instances, "to_frame"):
            query_df = query_instances.to_frame().T
        else:
            query_df = query_instances

        cf = self.dice.generate_counterfactuals(
            query_df,
            total_CFs=total_CFs,
            features_to_vary=features_to_vary,
            desired_class=desired_class
        )

        return cf

    def generate_batched(self, query_instances, predictions, features_to_vary, total_CFs=3):
        """
        Batch generate counterfactuals for a whole DataFrame, grouped by predicted class.

        For each unique predicted class, generates CFs that aim for the next higher class.

        :param query_instances: DataFrame of input features.
        :param predictions: array-like predictions (e.g., model.predict(X)).
        :param features_to_vary: list of feature names to vary.
        :param total_CFs: number of counterfactuals per input.
        :return: dict {desired_class: cf_examples_object}.
        """
        # Ensure predictions align with DataFrame index
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=query_instances.index)

        # Attach predictions to inputs for easy batching
        data_with_preds = query_instances.copy()
        data_with_preds["__y_pred__"] = self.dice_model.model.predict(query_instances)

        print("data with preds", data_with_preds)

        batches = {}
        unique_classes = sorted(data_with_preds["__y_pred__"].unique())

        for cls in unique_classes:

            if cls > 2:
                continue

            next_cls = int(cls) + 1

            a = data_with_preds[data_with_preds["__y_pred__"] == cls]
            b = a.drop(columns=["__y_pred__"])
            cls_batch = b

            print("predictions for batch 1", a['__y_pred__'].tolist())

            self.cls_batches[cls-1] = cls_batch

            if cls_batch.empty:
                continue

            print(f"\n=== Generating CFs for instances where predicted: {cls} -> desired: {next_cls} ===")
            print(cls_batch)
            cf = self.generate_and_show(
                query_instances=cls_batch,
                features_to_vary=features_to_vary,
                desired_class=next_cls,
                total_CFs=total_CFs
            )

            batches[next_cls] = cf

        return batches
