import unittest
import jax.numpy as jnp
from p_pack import circ, globals

class TestCirc(unittest.TestCase):

    def test_initialize_phases(self):
        """
        Test the phase initialization function.
        Checks if the returned phases have the correct shape.
        """
        depth = 5
        width = 8
        phases = circ.initialize_phases(depth, width)
        self.assertEqual(phases.shape, (depth, width // 2, 2))

    def test_layer_unitary(self):
        """
        Test the layer unitary creation.
        Checks if the output is a square matrix of the correct size and if it is unitary.
        """
        depth = 5
        width = 8
        all_phases = circ.initialize_phases(depth, width)
        layer_idx = 2
        unitary = circ.layer_unitary(all_phases, layer_idx)

        self.assertEqual(unitary.shape, (width, width))

        # Check for unitarity: U @ U.H == I
        identity = jnp.eye(width, dtype=jnp.complex64)
        product = unitary @ unitary.T.conj()
        self.assertTrue(jnp.allclose(product, identity, atol=1e-6))


    def test_data_upload(self):
        """
        Test the data upload mechanism.
        Checks if the returned unitary has the correct shape for a batch of data.
        """
        num_samples = 10
        feature_dim = 4
        data_set = jnp.ones((num_samples, feature_dim))
        unitary = circ.data_upload(data_set)
        self.assertEqual(unitary.shape, (num_samples, feature_dim * 2, feature_dim * 2))

    def test_measurement(self):
        """
        Test the measurement function.
        Provides a dummy unitary and checks the shapes of the returned values.
        """
        num_samples = 10
        num_modes = globals.num_modes_circ
        dummy_unitaries = jnp.array([jnp.eye(num_modes, dtype=jnp.complex64)] * num_samples)
        
        sub_unitaries, combos, probs, binary_probs = circ.measurement(dummy_unitaries)

        self.assertIsNotNone(sub_unitaries)
        self.assertIsNotNone(combos)
        self.assertEqual(probs.shape[0], num_samples)
        self.assertEqual(binary_probs.shape, (num_samples, 1))


if __name__ == '__main__':
    unittest.main()
