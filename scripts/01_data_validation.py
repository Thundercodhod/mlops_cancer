import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    """Loads breast cancer dataset, validates it, and logs to MLflow."""
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data as DataFrame
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame
        print("Data loaded successfully.")

        # 2. Perform validation checks
        num_rows, num_cols = df.shape
        num_classes = int(df["target"].nunique())
        missing_values = int(df.isnull().sum().sum())

        class_proportions = df["target"].value_counts(normalize=True)
        min_class_ratio = float(class_proportions.min())

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Class balance (min class ratio): {min_class_ratio:.4f}")

        # 3. Log metrics to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", min_class_ratio)
        mlflow.log_param("num_classes", num_classes)

        # Check validation status
        validation_status = "Success"
        if missing_values > 0 or num_classes != 2 or min_class_ratio < 0.20:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()