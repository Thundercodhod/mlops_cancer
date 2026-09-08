"""Data tests — ตรวจสอบความถูกต้องของชุดข้อมูล Breast Cancer"""

from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer(as_frame=True)
df = cancer_data.frame


def test_no_missing_values():
    """ตรวจสอบว่าไม่มีค่าสูญหายในชุดข้อมูล"""
    assert df.isnull().sum().sum() == 0, "พบค่าสูญหายในชุดข้อมูล"


def test_binary_classification_classes():
    """ตรวจสอบว่ามีจำนวนคลาสเท่ากับ 2 (Binary Classification)"""
    num_classes = df["target"].nunique()
    assert num_classes == 2, f"จำนวนคลาสต้องเท่ากับ 2 แต่พบ {num_classes}"


def test_class_balance_threshold():
    """ตรวจสอบว่าสัดส่วนของคลาสน้อยสุดไม่ต่ำกว่า 20%"""
    class_proportions = df["target"].value_counts(normalize=True)
    min_class_ratio = float(class_proportions.min())
    assert (
        min_class_ratio >= 0.20
    ), f"สัดส่วนคลาสน้อยสุด ({min_class_ratio:.4f}) ต่ำกว่า 20%"