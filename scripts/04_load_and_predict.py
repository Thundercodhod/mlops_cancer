import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    d = load_breast_cancer(as_frame=True)
    X, y = d.data, d.target
    for label in (0, 1):
        idx = y[y == label].index[0]
        sample = X.loc[[idx]]
        prediction = int(model.predict(sample)[0])
        actual_name = d.target_names[label]
        predicted_name = d.target_names[prediction]
        print(f"Actual: {actual_name} | Predicted: {predicted_name}"f" | Correct: {prediction == label}")
if __name__ == "__main__":
    load_and_predict()

