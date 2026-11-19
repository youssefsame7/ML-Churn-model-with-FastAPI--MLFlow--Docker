from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Evaluate XGboost model on testing data

    Args:
    model: trained xgboost
    X_test: Test features
    y_test: actual labels
    """

    preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
