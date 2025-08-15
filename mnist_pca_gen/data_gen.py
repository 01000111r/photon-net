

import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


out_dir = Path(__file__).resolve().parent / "mnist_pca_4"
out_dir.mkdir(exist_ok=True)

with np.load("mnist.npz") as f:
    X_tr_raw, y_tr = f["x_train"], f["y_train"]   # (60000, 28, 28), (60000,)
    X_te_raw, y_te = f["x_test"],  f["y_test"]    # (10000, 28, 28), (10000,)

# Merge, flatten to 784, and scale to [0,1]
X = np.concatenate([X_tr_raw, X_te_raw]).reshape(-1, 28*28).astype(np.float32) / 255.0
y = np.concatenate([y_tr, y_te]).astype(int)


# Filter digits 3 and 5  (your comment mentioned 7, but mask was [3,5])
mask = np.isin(y, [3, 5, 7, 9])
X = X[mask]
y = y[mask]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=0, stratify=y
)

# PCA projections for dimensions 2..20
for n in range(2, 21):
    pca = PCA(n_components=n, svd_solver="randomized", random_state=0)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    train_data = np.column_stack([X_train_pca, y_train])
    test_data = np.column_stack([X_test_pca, y_test])

    np.savetxt(out_dir / f"mnist_3-5-7-9_{n}d_train.csv", train_data, delimiter=',')
    np.savetxt(out_dir / f"mnist_3-5-7-9_{n}d_test.csv", test_data, delimiter=',')
    print(f"Saved PCA data for {n} dimensions")