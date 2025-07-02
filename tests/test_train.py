import unittest
import jax.numpy as jnp
from p_pack import train, globals, circ, pre_p

class TestTrain(unittest.TestCase):

    def test_train(self):
        """
        Test the main training function.
        This is a basic test that checks if the function runs without errors and returns
        outputs of the correct shape.
        """
        # Setup dummy data and parameters
        depth = 5
        feature_dim = 4
        num_samples = 10

        # Initialize phases and weights
        phases = circ.initialize_phases(depth, width=feature_dim*2)
        weights = jnp.ones((depth, feature_dim))

        # Create dummy data and labels
        data_set = jnp.ones((num_samples, feature_dim))
        labels = jnp.ones(num_samples)

        # Initialize optimizer moments
        m_phases = jnp.zeros_like(phases)
        v_phases = jnp.zeros_like(phases)
        m_weights = jnp.zeros_like(weights)
        v_weights = jnp.zeros_like(weights)

        # The 'init' tuple that the train function expects
        init = (phases, data_set, labels, weights, m_phases, v_phases, m_weights, v_weights)

        # Run the training function
        final_carry, loss_history = train.train(init)

        # Assertions
        self.assertEqual(len(final_carry), 8) # Check if all parts of the state are returned
        self.assertEqual(loss_history.shape, (globals.num_steps, 2)) # Check shape of the loss history

if __name__ == '__main__':
    unittest.main()
