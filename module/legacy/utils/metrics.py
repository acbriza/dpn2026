from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report
)


def calculate_specificity_youden(y_true, y_pred, pos_label=1):
    """
    Calculate specificity and Youden's index for binary classification.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, 1 - pos_label])

    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix does not have a binary shape. Ensure you are doing binary classification.")

    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    youden_index = sensitivity + specificity - 1

    return specificity, youden_index


def calculate_basic_metrics(y_true, y_pred, average='binary'):
    """
    Return a dictionary of basic performance metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def generate_classification_report(y_true, y_pred, target_names=None, output_dict=True):
    """
    Generate a classification report (optionally as a dict or string).
    """
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=output_dict)


def calculate_roc_auc(y_true, y_probs, average='macro', multi_class='ovr'):
    """
    Calculate ROC AUC score for binary or multi-class settings.

    Parameters:
    - y_true: Ground truth labels
    - y_probs: Probabilities or decision scores
    - average: 'macro', 'micro', 'weighted', etc.
    - multi_class: 'ovr' or 'ovo'
    """
    return roc_auc_score(y_true, y_probs, average=average, multi_class=multi_class)
