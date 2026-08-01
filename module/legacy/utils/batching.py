def get_batched_query_instances(model, X, max_class=3):
    """
    Groups a DataFrame into batches based on the model's predicted class,
    and returns a dict: {desired_class: DataFrame}

    Parameters:
        model : trained sklearn-like model with predict()
        X : pd.DataFrame, features only
        max_class : int, maximum class value in your ordinal task

    Returns:
        dict of {desired_class: DataFrame}
    """
    import pandas as pd

    # Predict classes
    preds = model.predict(X)

    # Attach predictions
    df_with_preds = X.copy()
    df_with_preds['pred_class'] = preds

    # Prepare output container
    grouped_batches = {}

    # Group by predicted class
    grouped = df_with_preds.groupby('pred_class')

    for pred_class, group_df in grouped:
        desired_class = pred_class + 1

        if desired_class > max_class:
            print(f"[SKIP] predicted={pred_class} → desired={desired_class} out of bounds.")
            continue

        # Drop the helper column so DiCE doesn't see it
        group_df = group_df.drop(columns=['pred_class'])

        grouped_batches[desired_class] = group_df

    return grouped_batches
