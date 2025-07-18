
import jax.numpy as jnp
from itertools import combinations
import itertools
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import factorial
import math


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

# out_state_combos = jnp.array(list(itertools.combinations_with_replacement(range(10), 3)))
# # parity_out_state_combos = jnp.sum(out_state_combos, axis=1) % 2
# print(out_state_combos[0:10])

combos = list(itertools.combinations_with_replacement(range(10), 5))
# repeats = []
# uniques = []
# count = []
# for combo in combos:
#     unique, counts = np.unique(combo, return_counts=True)
#     uniques.append(unique)
#     count.append(counts)
#     repeats.append(np.prod([math.factorial(int(c)) for c in counts]))


# print(combos[:10])
# print(uniques[:10])
# print(count[:10])
# print(repeats[:10])

#check to show that [0,0,0,1] counts the same as [1,0,0,0]
# print(combos[0:20])
# num_modes = 10
# input = '3'
# n = int(input)

# def input_config_maker(input : str, num_modes) -> list:
#     """
#     Converts a input string into an input coinfigurations
    
#     Args:
#         input_string (str): 'full' or 'n' or list of positions
                 
#     Returns:
#         list: A list of integers corresponding to the input string.
#     """
#     if input == 'full':
#         return [int(i) for i in range(num_modes)]
#     elif isinstance(input, list):
#         return np.array(input)
#     else:
#         n_gaps = n - 1
#         total_spaces = num_modes - n
#         av_spaces = (total_spaces)  // (n_gaps)
#         left_over = (total_spaces) % (n_gaps)
#         idx = 0
#         positions = [0]
#         for i in range(n_gaps):
#             empties = av_spaces + (1 if i < left_over else 0)
#             idx += empties + 1
#             positions.append(idx)
#         return positions
    
# print(input_config_maker([0,2,6], num_modes))

# import jax
# import jax.numpy as jnp
# from thewalrus import perm

# @jax.jit
# def permanent(A: jnp.ndarray) -> jnp.ndarray:
#     """
#     Compute the permanent of an n x n matrix A using Ryser's formula:
#       perm(A) = (-1)^n ∑_{S ⊆ [n]} (-1)^|S| ∏_{i=0..n-1} (∑_{j∈S} A[i,j])
#     Complexity: O(n · 2^n), fully static shapes for JIT.
    
#     Args:
#         A: shape (n, n)
#     Returns:
#         scalar permanent of A
#     """
#     n = A.shape[0]
#     # Total number of subsets of {0,...,n-1}
#     N = 1 << n  # 2^n

#     # Enumerate subsets by integer masks 0..2^n-1
#     subsets = jnp.arange(N)  # shape (N,)

#     # Build a (N, n) boolean array: masks[s, j] = 1 if j ∈ subset s
#     #   (subsets[:,None] >> jnp.arange(n)[None,:]) & 1  gives 0/1 bits
#     masks = ((subsets[:, None] >> jnp.arange(n)) & 1).astype(A.dtype)  # (N, n)

#     # For each subset s and each row i, compute sum_{j∈s} A[i,j]:
#     #   subsums[s, i] = ∑_j masks[s,j] * A[i,j]
#     subsums = jnp.einsum('sj,ij->si', masks, A)  # shape (N, n)

#     # For each subset s, the product over rows:
#     prods = jnp.prod(subsums, axis=1)            # shape (N,)

#     # Compute (-1)^{n - |S|} weights for each subset
#     pops   = jnp.sum(masks, axis=1)              # |S| for each subset
#     signs  = jnp.where(((n - pops) % 2) == 0, 1.0, -1.0)  # shape (N,)

#     # Ryser: perm(A) = (-1)^n * ∑_s [ signs[s] * prods[s] ]
#     return ((-1.0 ** n) * jnp.dot(signs, prods))

# def perm_3x3_jax(mat: jnp.array) -> float:
#     """
#     Calculates the permanent of a 3x3 matrix using JAX.
    
#     Args:
#         mat (jnp.array): A 3x3 matrix.

#     Returns:
#         float: The permanent of the matrix.
#     """
#     perms = jnp.array([
#         [0, 1, 2],
#         [0, 2, 1],
#         [1, 0, 2],
#         [1, 2, 0],
#         [2, 0, 1],
#         [2, 1, 0]
#     ])
#     return jnp.sum(jnp.prod(mat[jnp.arange(3), perms], axis=1))

# z = np.ones(shape=(3,3))
# print(perm_3x3_jax(z))
# print(permanent(z))
# print(perm(z))

import jax
from p_pack import circ
from p_pack import globals as g
from p_pack import  pre_p

presence_mask =  jnp.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
keep_probs_all = jnp.asarray([0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8])

key = jax.random.PRNGKey(2)

key, subkey = jax.random.split(key)
# 1) one uniform per mode
u = jax.random.uniform(subkey, shape=(10,))
# 2) survive if (was intended) & (u < keep_prob)
# survive = (presence_mask == 1) & (u < keep_probs_all)  # (NUM_MODES,) bool
survive = jnp.where(presence_mask == 1,
                u < keep_probs_all,   # if you wanted it
                False)                # otherwise
# 3) build a static survivors array
mode_idxs = jnp.arange(10, dtype=jnp.int32)
# sentinel = NUM_MODES → ensures dropped modes compare larger
survivors = jnp.where(survive, mode_idxs, 10)  
survivors = jnp.sort(survivors)            # real modes in front
# 4) count how many survived, goes through true/false list in survivor, truth =1
k = jnp.sum(survive).astype(jnp.int32)     # scalar in [0..NUM_MODES]
print(survivors, k, survive, u)


split_data = g.load_and_split_data(g.num_features)
train_set, train_labels, test_set, test_labels = split_data
# Rescale training + test features to range [-pi/2, pi/2]
data_set = pre_p.rescale_data(train_set, min_val = -(g.np.pi)/2, max_val = (g.np.pi/2))
test_set = pre_p.rescale_data(test_set, min_val = -(g.np.pi)/2, max_val = (g.np.pi/2))
phases = circ.initialize_phases(10, 2*g.num_features, )  
weights = g.jnp.ones(shape = [phases.shape[0],phases.shape[1]])
input_config = (presence_mask, keep_probs_all)


# Depth of the trainable part. 
depth = jax.lax.stop_gradient(phases).shape[0]
depth = int(depth)

# Please note that we broadcast over images in the data set. 
# The convention is that only the last two indices are used for matrix operations, 
# the others are broadcasting dimensions used for batches of images.

first_layers = circ.data_upload(weights[0,:]*data_set)
unitaries = first_layers
#print('First layer shape', first_layers)
#print('First layers shape', first_layers)
for layer in range(1,depth): 
    
    if (layer)%g.reupload_freq != 0: # every 'reupload_freq' layer is a upload layer 
        unitaries = circ.layer_unitary(phases, layer) @ unitaries
        #print('Layer', layer, 'shape', unitaries)
    # 'layer' is the layer index in the trainable part, starting from 0.
    else:        

        key = jax.random.PRNGKey(layer) 
        temp = jax.random.permutation(key, data_set.shape[1])
        temp = jax.lax.stop_gradient(temp)
        #temp = jnp.arange(data_set.shape[0])
        #shuffle all the images with the same permuatation, each reupload layer with a different permutation
        data_set_reupload = data_set[:,temp]
        
        #temp_permutation = data_set_reupload[:10, :3]
        #print(temp_permutation)


        unitaries_data_reupload = circ.data_upload(weights[layer,:]* data_set_reupload)
        #print('Reupload layer', layer, 'shape', unitaries_data_reupload)
        unitaries = unitaries_data_reupload @ unitaries



all_extracts, all_probs, binary_probs = \
    jax.lax.switch(1, circ.branch_fns, operand=(unitaries, survivors))

print(all_extracts[:10], all_probs[:10], binary_probs[:10])
print("calling switch with", k, type(k))
