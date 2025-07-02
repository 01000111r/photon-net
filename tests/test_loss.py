import unittest
import jax.numpy as jnp
from p_pack import loss, circ, model

class TestLoss(unittest.TestCase):

    def test_loss_calculation(self):
        """
        Test the loss function.
        This is a basic test to ensure the loss function returns a single scalar value.
        """
        # Setup
        depth = 5
        feature_dim = 4
        num_samples = 10

        phases = circ.initialize_phases(depth, width=feature_dim * 2)
        weights = jnp.ones((depth, feature_dim))
        data_set = jnp.ones((num_samples, feature_dim))
        labels = jnp.ones(num_samples)

        # We need to mock the prediction function used within the loss function
        # For a simple test, we can just check if the loss function runs.
        # A more advanced test would involve mocking model.predict_reupload
        try:
            loss_value = loss.loss(phases, data_set, labels, weights)
            self.assertTrue(jnp.isscalar(loss_value))
        except Exception as e:
            self.fail(f"loss.loss raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
