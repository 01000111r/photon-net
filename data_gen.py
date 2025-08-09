import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

out_dir = Path(__file__).resolve().parent / "mnist_pca_test"
out_dir.mkdir(exist_ok=True)

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data']
y = mnist['target'].astype(int)

# Filter digits 3, 5 and 7
mask = np.isin(y, [3, 5])
X = X[mask]
y = y[mask]

# Split into train/test maintaining class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=0, stratify=y
)

# Generate PCA projections for dimensions 2 through 20
for n in range(2, 21):
    pca = PCA(n_components=n, random_state=0)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    train_data = np.column_stack([X_train_pca, y_train])
    test_data = np.column_stack([X_test_pca, y_test])

    np.savetxt(out_dir / f"mnist_3-5_{n}d_train.csv", train_data, delimiter=',')
    np.savetxt(out_dir / f"mnist_3-5_{n}d_test.csv", test_data, delimiter=',')
    print(f"Saved PCA data for {n} dimensions")