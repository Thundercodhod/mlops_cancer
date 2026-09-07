from sklearn.datasets import load_breast_cancer
d = load_breast_cancer(as_frame=True)
print(d.frame.shape)
print(d.target_names)
print(d.frame['target'].value_counts(normalize=True))
