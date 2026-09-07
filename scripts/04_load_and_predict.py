import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction on Breast Cancer dataset.
    """
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # 1. โหลดโมเดลจาก Model Registry ด้วย Alias URI
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI."
        )
        return

    # 2. โหลดชุดข้อมูล Breast Cancer และดึงชื่อคลาส (['malignant', 'benign'])
    cancer_data = load_breast_cancer(as_frame=True)
    X = cancer_data.data
    y = cancer_data.target
    target_names = cancer_data.target_names  # 0: malignant, 1: benign

    # 3. ดึงตัวอย่างข้อมูลรายแรกของแต่ละคลาส (คลาส 0 และ คลาส 1)
    idx_class_0 = y[y == 0].index[0]  # ดึงแถวแรกที่เป็น malignant
    idx_class_1 = y[y == 1].index[0]  # ดึงแถวแรกที่เป็น benign
    sample_indices = [idx_class_0, idx_class_1]

    sample_X = X.loc[sample_indices]
    actual_y = y.loc[sample_indices].values

    # 4. นำโมเดลมาทำนายผล
    predictions = model.predict(sample_X)

    # 5. แสดงผลการทำนาย เปรียบเทียบกับคำจริงพร้อมชื่อคลาส (malignant / benign)
    print("=" * 55)
    for i, idx in enumerate(sample_indices):
        actual_num = actual_y[i]
        pred_num = int(predictions[i])

        actual_label = target_names[actual_num]
        pred_label = target_names[pred_num]

        is_correct = (
            "ทำนายถูกต้อง (Match)"
            if actual_num == pred_num
            else "ทำนายไม่ถูกต้อง (Mismatch)"
        )

        print(f"Sample #{i + 1} (Row Index: {idx}):")
        print(f"  Actual Class    : {actual_label} (Code: {actual_num})")
        print(f"  Predicted Class : {pred_label} (Code: {pred_num})")
        print(f"  Result          : {is_correct}")
        print("-" * 55)


if __name__ == "__main__":
    load_and_predict()