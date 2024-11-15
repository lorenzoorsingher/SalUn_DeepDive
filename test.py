import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def evaluate_unlearning(model, X_test, y_test, test_idx, class_to_forget=0):
    """
    Evaluates the effectiveness of unlearning on a specific class.

    Args:
        model: The trained machine learning model after unlearning.
        X_test: The test set features.
        y_test: The test set labels.
        test_idx: The indices of the test set that are relevant for evaluation.
        class_to_forget: The class label that the model should have unlearned.

    Returns:
        A dictionary containing the overall accuracy and the accuracy on the 
        forget class.
    """

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Predict labels on the test set with a progress bar
    y_pred = []
    with torch.no_grad():
        for i in tqdm(range(len(X_test)), desc="Evaluating"):
            output = model(X_test[i].unsqueeze(0))
            _, pred = torch.max(output, 1)
            y_pred.append(pred.item())

    y_pred = torch.tensor(y_pred).to(device)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())

    # Identify indices of the forget class in the test set
    forget_idx = np.where(y_test.cpu() == class_to_forget)[0]
    retain_idx = np.where(y_test.cpu() != class_to_forget)[0]

    # Calculate accuracy on the forget class
    forget_accuracy = accuracy_score(y_test.cpu()[forget_idx], y_pred.cpu()[forget_idx])
    retain_accuracy = accuracy_score(y_test.cpu()[retain_idx], y_pred.cpu()[retain_idx])

    return {
        "overall_accuracy": overall_accuracy,
        "forget_accuracy": forget_accuracy,
        "retain_accuracy": retain_accuracy,
    }