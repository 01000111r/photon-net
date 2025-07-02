# photonic_classifier/circ.py

import numpy as np
import jax
import jax.numpy as jnp
from numpy.random import default_rng
from functools import partial
from jax.scipy.special import factorial
import itertools
from thewalrus import perm
from p_pack import globals
import math


# global constants
rng = default_rng(1337)
reupload_freq = globals. reupload_freq
num_modes_circ = globals.num_modes_circ

# This is the function used for loss calculation /predictions etc
# we need to inlcude only trainable phases here
# Considering alterante layers of data reuploading 

def initialize_phases(depth: int, width: int = None, mask: np.ndarray = None) -> jnp.array:
    """
    Initializes the phase parameters for the photonic circuit.å

    The phases are initialized with small random values to avoid barren plateaus.
    A mask can be provided to fix certain phases to zero, making them non-trainable.
    By default, data-uploading layers (determined by `reupload_freq`) are masked out.

    Args:
        depth (int): The number of layers in the circuit.
        width (Optional[int]): The number of modes in the circuit. Defaults to `depth`.
        mask (Optional[np.ndarray]): A binary mask to apply to the phases.
                                     A value of 0 freezes a phase.

    Returns:
        jnp.array: A JAX array of initialized phases.
    """
    # Default case is Clements et al. layout, with all beam splitters tunable.
    if width == None:
        width = depth 
    if mask == None:
        mask = np.ones(shape = [depth, width//2, 2])
        #mask = np.zeros((depth, width // 2, 2))
        for i in range(0,depth, reupload_freq):  # every reupload_freq-th layer is a uploading layer 
            mask[i] = 0

    # // 2 is integer division by 2, including rounding down.
    # The last two says that these two phases belong  to the same beamsplitter.
    # That is also why we divide the width by 2.
    phases = rng.uniform( low = -0.1, high = 0.1 , size =  [depth, width//2, 2])
    # The mask allows to set some phases to zero. This can be used if one wants to 
    # fix some beam splitters to the identity, for example for modularity.
    phases = mask*phases   
    phases = jnp.array(phases)
    return phases


# JAX likes to complain A LOT. Therefore, we need to explicitly declare variables as static.
# E.g. JAX doesn't like to use shapes of trainable arrays to define new arrays, or anything really.     
#@jax.jit
@partial(jax.jit, static_argnames=['layer'])
def layer_unitary(all_phases: jnp.array, layer: int, mask: jnp.array = None) -> jnp.array:
    """
    Constructs the unitary matrix for a single trainable layer of the circuit.

    Args:
        all_phases (jnp.array): The full tensor of phase parameters for all layers.
        layer (int): The index of the layer to construct the unitary for.
        mask (Optional[jnp.array]): An optional mask to apply to the layer's phases.

    Returns:
        jnp.array: The complex-valued unitary matrix for the specified layer.
    """
    #layer = jax.lax.stop_gradient(layer) # doesn't work, don't ask me why.
    
    width = 2*jax.lax.stop_gradient(all_phases).shape[1] 
    # Stopping the gradient here allows to use the size of an input tensor to define other tensors.
    # Depth of the trainable part.
    depth = jax.lax.stop_gradient(all_phases).shape[0]
    if mask == None:
        # The default mask allows all phases to be trained
        mask = jnp.ones(shape = [depth, width//2])
    trainable_layer_phases = jnp.zeros(shape = [width//2 , 2])
    # Notice that the scalar operation '*' is applied for each beamsplitter of the layer individually, 
    # hidden in the fact that mask[layer] still has a dimension for the width, and the ':'.

    # below retrieves a given layer's phases from the all_phases tensor
    trainable_layer_phases = trainable_layer_phases.at[:, 0].set(mask[layer]*all_phases[layer,:,0])
    trainable_layer_phases = trainable_layer_phases.at[:, 1].set(mask[layer]*all_phases[layer,:,1])
    
    unitary = jnp.zeros(shape = [width, width], dtype = jnp.complex64)
    # Odd layers get an offset of one for placing beamsplitters.
    
    offset = (layer) % 2
    # Take care of wires that do not see a beamsplitter in this layer.
    if offset == 1:
        unitary = unitary.at[0,0].set(1.0)
        # If the width is even and the offset is one, also the last wire does not get a beamsplitter.
        if width % 2 == 0:
            # -1 gives the last entry
            unitary = unitary.at[-1, -1].set(1.0)
    else:  # Offset is 0, so for odd number of wires the last one cannot get a beamsplitter.  
        if width % 2 == 1:
            unitary = unitary.at[-1,-1].set(1.0)  
            
    # Now, write the actual layers
    # Since the entries look so different, I have no clever idea for how to vectorize/broadcast this...
    for index in range( (width-offset)//2):
        p = trainable_layer_phases[index,0]
        q = trainable_layer_phases[index,1]
        # Taken from old code. However, for p=q=0, it does not give the identity. Therefore the mask doesn't work.
        #unitary = unitary.at[offset+2*index, offset+2*index].set(jnp.exp(p*1j)*jnp.sin(q/2))
        #unitary = unitary.at[offset+2*index , offset+2*index+1].set(jnp.cos(q/2))
        #unitary =  unitary.at[offset+2*index+1, offset+2*index].set(jnp.exp(p*1j)*jnp.cos(q/2))
        #unitary = unitary.at[offset+2*index+1, offset+2*index+1].set(-jnp.sin(q/2))
        
        # To get the mask to work, I am using a different parameterization.
        unitary = unitary.at[offset + 2*index,  offset + 2*index].set(0.5*(1 + jnp.exp(1j*p)))
        unitary = unitary.at[offset + 2*index,  offset + 2*index + 1].set(0.5*(jnp.exp(1j*q) - jnp.exp(1j*(q+p))))
        unitary = unitary.at[offset + 2*index + 1, offset + 2*index].set(0.5*(1 - jnp.exp(1j*p)))
        unitary = unitary.at[offset + 2*index + 1, offset + 2*index + 1].set(0.5*(jnp.exp(1j*q) + jnp.exp(1j*(q+p))))

       #need to create a diagram to illustrate how this unitary is made, need to comfirm how this is


    # Taken from old code to remind myself.
    #splitters = jnp.array([[jnp.exp(p*1j)*np.sin(q/2), np.cos(q/2)], [np.exp(p*1j)*np.cos(q/2), -np.sin(q/2)]] for q,p in [q_all, p_all])

    return unitary

@jax.jit
def data_upload(data_set: jnp.array) -> jnp.array:
    """
    Constructs the unitary matrices for the data uploading layer.

    This function creates a batch of block-diagonal unitary matrices, where each
    matrix encodes one sample (e.g., an image) from the input data set.

    Args:
        data_set (jnp.array): The input data, with shape (num_samples, num_features).

    Returns:
        jnp.array: A batch of unitary matrices with shape (num_samples, width, width).
    """
    num_samples = jax.lax.stop_gradient(data_set).shape[0]

    # Each pixel gets its BS, therefore factor 2 for counting overall system width
    width = 2*jax.lax.stop_gradient(data_set).shape[1]
    # Again, the 3rd dimension with 2 represents the two phases for each beamsplitter. 

    # is this the fastest way to fill the array with what we want or ist here a faster way
    phases = (jnp.pi/2)*jnp.ones(shape = [num_samples, width//2, 2]) 
    # The first of the phases of the beam splitters are set to be the feature values, the second phases are set 
    # to a constant pi/2 . 0 doesn't work because minimal and maximal pixel brightness act on the uniform superposition
    # state identically, with the parameterization below. For q = pi/2, minimal and maximal pixel brightness move the
    # uniform superposition into orthogonal states.  
    phases = phases.at[:,:,0].set(data_set)

    # Note that our "unitary" has 3 dimensions. The 1st dimension is a batching dimension, 
    # representing the index of the image. This allows to parallelize the calculation of hthe loss 
    # over the full training set later.
    
    unitary = jnp.zeros(shape = [num_samples, width, width], dtype = jnp.complex64)    
    for index in range( width//2 ):    
        #print('yes')
        p = phases[:, index, 0]
        q = phases[:, index, 1]
        # Note that p and q are 1-dimensional tensors here. We use that all operations here like jnp.exp are 
        # applied entry-by-entry to calculate the uploading unitary for all images in parallel.
        # That means each entry of p and q corresponds to one image, which corresponds to 
        # one entry in the : in dimension 0. 
        unitary = unitary.at[:,2*index, 2*index].set(0.5*(1+jnp.exp(1j*p)))
        unitary = unitary.at[:,2*index , 2*index+1].set(0.5*(jnp.exp(1j*q)-jnp.exp(1j*(q+p))))
        unitary = unitary.at[:,2*index+1, 2*index].set(0.5*(1-jnp.exp(1j*p)))
        unitary = unitary.at[:,2*index+1, 2*index+1].set(0.5*(jnp.exp(1j*q)+jnp.exp(1j*(q+p))))

    return unitary

import jax, jax.numpy as jnp

@jax.jit
def data_upload_v2(data_set: jnp.ndarray) -> jnp.ndarray:
    """
    A vectorized implementation to construct the data uploading unitary matrices.

    This version avoids loops for better performance on hardware accelerators.

    Args:
        data_set (jnp.ndarray): Input data of shape (batch, n_pixels).

    Returns:
        jnp.ndarray: Block-diagonal upload unitaries of shape (batch, 2*n_pixels, 2*n_pixels).
    """
    num_samples, n_pixels = data_set.shape
    width          = 2 * n_pixels                          
    q              = jnp.pi / 2                            # constant phase for one of the values of Bs pair

    # ---------------------------------------------------------------------
    # 2.  Pre-compute the four complex numbers that define each 2×2 block
    # ---------------------------------------------------------------------
    p         = data_set                                   # (B, n_pixels)
    exp_ip    = jnp.exp(1j * p)
    exp_iq    = jnp.exp(1j * q)                            # scalar → broadcast
    exp_iqp   = jnp.exp(1j * (p + q))

    a = 0.5 * (1.0 + exp_ip)                               # (B, n_pixels)
    b = 0.5 * (exp_iq - exp_iqp)
    c = 0.5 * (1.0 - exp_ip)
    d = 0.5 * (exp_iq + exp_iqp)

    #  (B, n_pixels, 2, 2) – each pixel’s 2×2 beamsplitter
    blocks = jnp.stack(
        [jnp.stack([a, b], axis=-1),                       # row 0
         jnp.stack([c, d], axis=-1)],                     # row 1
        axis=-2
    )

    # ---------------------------------------------------------------------
    # 3.  Assemble block-diagonal matrices in one shot
    # ---------------------------------------------------------------------
    eye = jnp.eye(n_pixels, dtype=blocks.dtype)            # (n, n)

    # Broadcast:   (B,  n,  2, 2)  ->  (B, n, n, 2, 2)  with zeros off the diagonal
    diag_blocks = blocks[:, :,  None] * eye[None, :, :, None, None]

    # Re-shape to (B, 2n, 2n)
    unitary = (diag_blocks
               .reshape(num_samples, n_pixels, n_pixels, 2, 2)
               .transpose(0, 1, 3, 2, 4)                   # (B, n, 2, n, 2)
               .reshape(num_samples, width, width)
               .astype(jnp.complex64))

    return unitary



# this can be faster and also generalise
def perm_3x3_jax(mat: jnp.array) -> float:
    """
    Calculates the permanent of a 3x3 matrix using JAX.
    
    Args:
        mat (jnp.array): A 3x3 matrix.

    Returns:
        float: The permanent of the matrix.
    """
    perms = jnp.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0]
    ])
    return jnp.sum(jnp.prod(mat[jnp.arange(3), perms], axis=1))


#Ideally this should be inside the measurement function but it clashes with JAX.
# it is used in the factorials calculation (denominator in the probabilities of bunched outcomes)
out_state_combos = jnp.array(list(itertools.product(range(num_modes_circ), repeat=3))) 

def repeats_factorials(num_modes, num_photons=3):
    combos = list(itertools.product(range(num_modes), repeat=num_photons))
    repeats = []
    for combo in combos:
        unique, counts = np.unique(combo, return_counts=True)
        repeats.append(np.prod([math.factorial(int(c)) for c in counts]))
    return jnp.array(repeats) 


# @partial(jax.jit, static_argnames=['num_modes']) #recompiles when a new num_modes is passed
def measurement(unitaries: jnp.array, num_photons: int = 3) -> tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Simulates the measurement process of the photonic circuit.

    It calculates the probabilities of detecting photons in different output modes,
    computes the permanents of submatrices, and aggregates these into binary
    classification probabilities (+1 or -1) based on the parity of output modes.

    Args:
        unitaries (jnp.array): The batch of final unitary matrices from the circuit.
        num_photons (int): The number of photons in the input state. Defaults to 3.
        factorials (jnp.array): Pre-computed factorial values for probability calculation.

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]: A tuple containing:
            - all_extracts: The submatrices used for permanent calculation.
            - out_state_combos: All possible output state combinations.
            - all_probs: The raw probability for each output combination.
            - binary_probs: The final aggregated probability for the +1 outcome.
    """
    n = unitaries.shape[0]
    num_modes = unitaries.shape[1]
    #out_state_combos = jnp.array(list(combinations(range(num_modes), num_photons)))
    
    out_state_combos = jnp.array(list(itertools.product(range(num_modes), repeat=num_photons)))
    n_combos = out_state_combos.shape[0]

    factorials_dyn = repeats_factorials(num_modes=unitaries.shape[-1],
                                    num_photons=num_photons)

    parity_out_state_combos = jnp.sum(out_state_combos, axis=1) % 2

    # Truncate to first 3 columns
    #print('Unitaries', unitaries[:5, :, :])
    input_state_modes = jnp.array([0, num_modes//2, num_modes-1])
    # photons are always in the first, middle and last modes
    unitaries_truncated = unitaries[:, :, input_state_modes]

    
    # Vectorized extraction of submatrices for all samples and all combos
    def extract_submatrices(unitary):
        # unitary: (num_modes, 3)
        # out_state_combos: (n_combos, 3)
        return unitary[out_state_combos, :]  # (n_combos, 3, 3)

    # Apply to all samples--
    all_extracts = jax.vmap(extract_submatrices)(unitaries_truncated)  # (n, n_combos, 3, 3)


    

    # Vectorized permanent calculation over all submatrices
    perm_fn = jax.vmap(lambda mat: jnp.abs(perm_3x3_jax(mat))**2)
    all_probs0 = jax.vmap(perm_fn, in_axes=0)(all_extracts)  # (n, n_combos)
    #print('Truncated unitaries', unitaries_truncated[0, :, :])
    #print('All probs0', all_probs0[:,:10])
    #print("Combo :", out_state_combos[:10])
    #print("Sample submatrix for combo 216:", all_extracts[0, :5])
    #print("Permanent for sample:", perm_3x3_jax(all_extracts[0, :5]))
    all_probs = all_probs0 / factorials_dyn  # Broadcasting over columns
    #print('All probs', all_probs.shape)
    # Now, for each row in all_probs, every column is divided by the corresponding factorial.
    ##
    plus_1_probs = all_probs * parity_out_state_combos
    plus_minus1_probs = all_probs * (1 - parity_out_state_combos)
    #print('Plus 1 probs', plus_1_probs[:10, :5])
    # Sum over all output state probabilities for each sample
    total_probs = jnp.sum(all_probs, axis=1, keepdims=True)  # shape (n, 1)
    #print('Total probs', total_probs)
    #need to verify if total_probs is correct

    # Normalise all probabilities
    all_probs_norm = all_probs / total_probs
    plus_1_probs_norm = plus_1_probs / total_probs
    plus_minus1_probs_norm = plus_minus1_probs / total_probs

    # Normalised binary probabilities
    binary_probs = jnp.sum(plus_1_probs_norm, axis=1, keepdims=True)
    binary_probs_minus = jnp.sum(plus_minus1_probs_norm, axis=1, keepdims=True)

    # Optionally print to check
    #print('Sum of all_probs_norm (should be 1):', jnp.sum(all_probs_norm, axis=1))
    #print('Sum of binary_probs + binary_probs_minus (should be 1):', (binary_probs + binary_probs_minus).squeeze())
    
    ##

    
    # 0 = even (-1), 1 = odd (+1)
    #even_indices = jnp.where(parity_out_state_combos == 0)[0]
    
    #plus_1_probs = all_probs*parity_out_state_combos
    #plus_minus1_probs = all_probs*(1-parity_out_state_combos)
    #print(plus_1_probs[:10, :5])
    #binary_probs = jnp.sum(plus_1_probs, axis=1, keepdims=True)
    #binary_probs_minus = jnp.sum(plus_minus1_probs, axis=1, keepdims=True)


    #total_probs = binary_probs + binary_probs_minus
    #print('total_probs', total_probs[:10])
    #print('minus', binary_probs_minus[:10])  
    #print('plus',binary_probs[:10])
    #print('plus total sum', jnp.sum(binary_probs))
    #print('minus total sum', jnp.sum(binary_probs_minus))
  
    return all_extracts, out_state_combos, all_probs, binary_probs