
import jax.numpy as jnp
from itertools import combinations
import itertools
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import factorial


# depth =3
# width = 4
# mask = np.zeros((depth, width //2, 2))
# for i in range(1, depth, 2):  # odd layers only
#     mask[i] = 1.0
# print(mask)

# # Example usage
# out_state_combos = jnp.array(list(itertools.product(range(10), repeat=3)))
# unitary = np.array([
#     [10, 11, 12],  # Row 0
#     [20, 21, 22],  # Row 1
#     [30, 31, 32],  # Row 2
#     [40, 41, 42],  # Row 3
#     [50, 51, 52]   # Row 4
# ])

# def extract_submatrices(unitary):
#     # unitary: (num_modes, 3)
#     # out_state_combos: (n_combos, 3)
#     return unitary[out_state_combos[:5], :] 

# print('Output state combinations:', out_state_combos[:5])

# extracted_submatrices = extract_submatrices(unitary)
# print('Extracted submatrices:', extracted_submatrices[:10])

# z = np.ones(shape=(2, 3, 4))
# z = [[[1, 1, 1, 1],
#   [1, 1, 1, 1],
#   [1, 1, 1, 1]],

#  [[1, 1, 1, 1],
#   [1, 1, 1, 1],
#   [1, 1, 1, 1]]]
# # print([i for i in range(1,10)])

# depth_t = 10
# reupload_freq_t = 3

# # print("mask layer indexes for phase tensor:",  [i for i in range(0,depth_t, reupload_freq_t)])

# # print("layer indicies for trainable phases", [i for i in range(1,depth_t) if (i)%reupload_freq_t != 0])

# key = jax.random.PRNGKey(3) 
# temp = jax.random.permutation(key, z.shape[1])

# #temp = jnp.arange(data_set.shape[0])
# z_new= z[:,temp]

# print(z)



# b = np.array([0,1,1])
# unique, counts = jnp.unique(b, return_counts=True)
# repeats = counts[counts > 1]
# #return repeats.sum() #if repeats.size > 0 else 0
# print(repeats) #if repeats.size > 0 else 0

out_state_combos = jnp.array(list(itertools.combinations_with_replacement(range(10), 3)))
# parity_out_state_combos = jnp.sum(out_state_combos, axis=1) % 2
print(out_state_combos[0:10])