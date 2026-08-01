# models/model_factory.py

from mord import LogisticAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Example PyTorch model placeholder
# from your_pytorch_models import CoralNN

def get_model(model_type, backend, x_train, y_train, preprocessor=None, **kwargs):
    """
    Factory function to initialize and fit the model.

    :param preprocessor:
    :param model_type: str, e.g. 'logisticat', 'coral'
    :param backend: str, e.g. 'sklearn', 'pytorch'
    :param x_train: training features
    :param y_train: training labels
    :return: trained model instance
    """

    if backend == "sklearn":
        if model_type == "logisticat":
            print("Building MORD LogisticAT model...")

            pipeline = Pipeline(steps=[
                ("transformer", preprocessor),
                ("classifier", LogisticAT())
            ])
            pipeline.fit(x_train, y_train)
            return pipeline

        elif model_type == "logistic":
            print("Building standard Logistic Regression model...")
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression())
            ])
            pipeline.fit(x_train, y_train)
            return pipeline

        elif model_type == "randomforest":
            print("Building standard Logistic Regression model...")
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("logistic", RandomForestClassifier())
            ])
            pipeline.fit(x_train, y_train)
            return pipeline

        else:
            raise ValueError(f"Unsupported sklearn model type: {model_type}")

    elif backend == "pytorch":
        if model_type == "coral":
            print("Building PyTorch CORAL ordinal regression model...")
            # coral_model = CoralNN(input_dim=X_train.shape[1], **kwargs)
            # coral_model.train(X_train, y_train)  # You'd define your training loop
            # return coral_model
            raise NotImplementedError("PyTorch CORAL model placeholder — implement your training loop here.")
        else:
            raise ValueError(f"Unsupported PyTorch model type: {model_type}")

    else:
        raise ValueError(f"Unsupported backend: {backend}")
