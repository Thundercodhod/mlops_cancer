import os  # เพิ่ม import นี้สำหรับจัดการ file path
import sys
 
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient  # ใช้สำหรับตั้ง Alias ของโมเดล (MLflow 3)
from mlflow.artifacts import download_artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
 
def train_evaluate_register(preprocessing_run_id, C=1.0):
    """
    Loads preprocessed data, trains a model, evaluates it, and
    registers the model in the MLflow Model Registry if it meets
    the performance threshold.
    """
    ACCURACY_THRESHOLD = 0.95
    AUC_THRESHOLD = 0.98
    MODEL_NAME = "cancer-classifier-prod"
    mlflow.set_experiment("Cancer - Model Training")
 
    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
 
        # 1. โหลดข้อมูลจาก Artifacts ของ Preprocessing Run
        try:
            # --- START: ดาวน์โหลด artifact มาที่ local ก่อนอ่าน (ใช้ได้ทุกระบบปฏิบัติการ) ---
            # 1.1 ใช้ MLflow ดาวน์โหลด Artifacts ลงมาที่ local path ชั่วคราว
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")
 
            # 1.2 สร้างพาธไปยังไฟล์ CSV ที่ดาวน์โหลดมา
            train_path = os.path.join(local_artifact_path, "train.csv")
            test_path = os.path.join(local_artifact_path, "test.csv")
 
            # 1.3 อ่านไฟล์ CSV จาก local path ที่ถูกต้อง
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Successfully loaded data from downloaded artifacts.")
            # --- END: ดาวน์โหลด artifact ---
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Please ensure the preprocessing_run_id is correct.")
            sys.exit(1)
 
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
 
        # 2. สร้าง Scikit-learn Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)
 
        # 3. ประเมินผลโมเดล
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        print(f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")
        
 
        # 4. Log Parameters, Metrics, และ Model (Pipeline)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        # MLflow 3: ใช้ name= แทน artifact_path= (ที่เลิกใช้แล้ว)
        # และแนบ input_example เพื่อให้ MLflow สร้าง model signature ให้อัตโนมัติ
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="cancer_classifier_pipeline",
            input_example=X_train.head(5)
        )
 
        # 5. ตรวจสอบและลงทะเบียนโมเดล
        if acc >= ACCURACY_THRESHOLD and auc >= AUC_THRESHOLD:
            print(f"Model accuracy ({acc:.4f}) meets the threshold. Registering model...")
            # MLflow 3: ใช้ model_info.model_uri (รูปแบบ models:/<model_id>) ลงทะเบียนได้เลย
            registered_model = mlflow.register_model(model_info.model_uri, MODEL_NAME)
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
 
            # MLflow 3 ยกเลิก Model Stage (Staging/Production) แล้ว ให้ใช้ Alias แทน
            client = MlflowClient()
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="staging",
                version=registered_model.version
            )
            print(f"Set alias '@staging' -> {MODEL_NAME} version {registered_model.version}")
            print(f"Registered version {registered_model.version} @staging")
        else:
            print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")
            print("Gate failed. Not registering.")
        print("Training run finished.")
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C_value]")
        sys.exit(1)
 
    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)

