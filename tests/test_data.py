from sklearn.datasets import load_breast_cancer
df = load_breast_cancer(as_frame=True).frame
def test_schema():
    assert df.shape == (569, 31)
    assert "target" in df.columns
def test_no_missing():
    assert df.isnull().sum().sum() == 0
def test_two_classes():
    assert set(df["target"].unique()) == {0, 1}
def test_class_balance():
    assert df["target"].value_counts(normalize=True).min() >= 0.20
def test_radius_range():
    assert df["mean radius"].between(0, 40).all()