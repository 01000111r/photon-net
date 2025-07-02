import unittest
import jax.numpy as jnp
from p_pack import model, circ

class TestModel(unittest.TestCase):

    def test_full_unitaries_data_reupload(self):
        """
        Test the function that builds the full unitary for the model.
        Checks the shapes of the outputs.
        """
        depth = 5
        feature_dim = 4
        num_samples = 10

        phases = circ.initialize_phases(depth, width=feature_dim * 2)
        weights = jnp.ones((depth, feature_dim))
        data_set = jnp.ones((num_samples, feature_dim))

        outputs = model.full_unitaries_data_reupload(phases, data_set, weights)
        
        # Expecting 4 outputs
        self.assertEqual(len(outputs), 4)

        unitaries, sub_unitaries, label_probs, binary_probs_plus = outputs
        
        self.assertEqual(unitaries.shape[0], num_samples)
        self.assertIsNotNone(sub_unitaries)
        self.assertEqual(label_probs.shape[0], num_samples)
        self.assertEqual(binary_probs_plus.shape, (num_samples, 1))


    def test_predict_reupload(self):
        """
        Test the prediction function.
        Checks the shapes of the outputs.
        """
        depth = 5
        feature_dim = 4
        num_samples = 10

        phases = circ.initialize_phases(depth, width=feature_dim * 2)
        weights = jnp.ones((depth, feature_dim))
        data_set = jnp.ones((num_samples, feature_dim))

        probs, adjusted_binary_probs = model.predict_reupload(phases, data_set, weights)

        self.assertEqual(probs.shape[0], num_samples)
        self.assertEqual(adjusted_binary_probs.shape, (num_samples, 1))

if __name__ == '__main__':
    unittest.main()
