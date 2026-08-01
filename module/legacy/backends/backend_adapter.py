# backends/backend_adapter.py

import dice_ml
from dice_ml.utils import helpers
import pandas as pd


def get_dice_data(x_train, y_train, input_features, continuous_features, target):
    """
    Build DiCE Data Interface.

    Returns:
      dice_ml.Data object
    """
    print("Building DiCE Data Interface...")

    # Combine X and y to one DataFrame
    df = pd.concat([x_train, y_train], axis=1)
    df.columns = input_features + [target]

    data_interface = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        outcome_name=target,
        outcome_type="ordinal"
    )

    return data_interface


def get_dice_model(model, backend):
    """
    Build DiCE Model Interface.

    Returns:
      dice_ml.Model object
    """
    print(f"Building DiCE Model Interface for backend: {backend}...")

    if backend == "sklearn":
        dice_model = dice_ml.Model(model=model, backend="sklearn")
    elif backend == "pytorch" or backend == "PYT":
        dice_model = dice_ml.Model(model=model, backend="PYT")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return dice_model


def get_dice_components(model, backend, x_train, y_train,
                        input_features, continuous_features, target):
    """
    Full Adapter: returns (dice_model, dice_data)
    """
    dice_data = get_dice_data(
        x_train, y_train, input_features, continuous_features, target
    )
    dice_model = get_dice_model(model, backend)
    return dice_model, dice_data
