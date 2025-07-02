import unittest
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
from p_pack import pre_p

class TestPreP(unittest.TestCase):

    def test_rescale_data(self):
        """
        Test the data rescaling function.
        Checks if the data is correctly scaled to the target range.
        """
        data = jnp.array([-10., 0., 10.])
        min_val = -np.pi / 2
        max_val = np.pi / 2
        rescaled = pre_p.rescale_data(data, min_val, max_val)
        
        self.assertTrue(jnp.all(rescaled >= min_val))
        self.assertTrue(jnp.all(rescaled <= max_val))

    def setUp(self):
        """Create a dummy CSV file for testing load_mnist_35."""
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.feature_dim = 4
        self.fname = f"mnist_3-5_{self.feature_dim}d_train.csv"
        self.path = os.path.join(self.test_dir, self.fname)
        
        # Create dummy data: 5 samples, 4 features + 1 label
        data = np.random.rand(5, self.feature_dim + 1)
        df = pd.DataFrame(data)
        df.to_csv(self.path, index=False)

    def tearDown(self):
        """Remove the dummy CSV and directory after tests."""
        os.remove(self.path)
        os.rmdir(self.test_dir)

    def test_load_mnist_35(self):
        """
        Test loading data from a CSV file.
        Checks if the data and labels are loaded with the correct shapes.
        """
        X, y = pre_p.load_mnist_35(self.test_dir, self.feature_dim, split="train")
        
        self.assertEqual(X.shape, (5, self.feature_dim))
        self.assertEqual(y.shape, (5,))

if __name__ == '__main__':
    unittest.main()
