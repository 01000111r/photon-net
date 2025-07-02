import unittest
import jax.numpy as jnp
from p_pack import optimiser, circ

class TestOptimiser(unittest.TestCase):

    def test_adam_step(self):
        """
        Test a single step of the Adam optimizer.
        Checks if the function executes and returns outputs of the correct shape.
        """
        # Setup
        depth = 5
        feature_dim = 4
        num_samples = 10

        params_phases = circ.initialize_phases(depth, width=feature_dim * 2)
        params_weights = jnp.ones((depth, feature_dim))
        data_set = jnp.ones((num_samples, feature_dim))
        labels = jnp.ones(num_samples)
        
        m_phases = jnp.zeros_like(params_phases)
        v_phases = jnp.zeros_like(params_phases)
        m_weights = jnp.zeros_like(params_weights)
        v_weights = jnp.zeros_like(params_weights)

        carry = [params_phases, data_set, labels, params_weights, m_phases, v_phases, m_weights, v_weights]
        step_number = 1

        new_carry, loss_info = optimiser.adam_step(carry, step_number)

        self.assertEqual(len(new_carry), 8)
        self.assertEqual(loss_info.shape, (2,))

if __name__ == '__main__':
    unittest.main()
