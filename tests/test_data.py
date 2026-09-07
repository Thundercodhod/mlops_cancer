"""Model tests — model gate ที่บล็อกไม่ให้โมเดลแย่ ๆ หลุดเข้า main"""
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
MIN_ACCURACY = 0.95
 
X, y = load_wine(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=10.0, max_iter=10000, random_state=42)),
]).fit(X_tr, y_tr)
 
 
def test_accuracy_gate():
    acc = accuracy_score(y_te, pipe.predict(X_te))
    assert acc >= MIN_ACCURACY, f"accuracy {acc:.4f} ต่ำกว่าเกณฑ์ {MIN_ACCURACY}"
 
 
def test_pipeline_has_scaler():
    """preprocessing ต้องอยู่ใน pipeline เดียวกับโมเดล — กัน Training-Serving Skew"""
    assert "scaler" in pipe.named_steps
 
 
def test_predict_shape():
    assert pipe.predict(X_te.head(5)).shape == (5,)
