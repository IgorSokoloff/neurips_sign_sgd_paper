import numpy as np
import time
import os

from mpi4py import MPI
from numpy.random import normal
from numpy.linalg import norm

s = 1

def block_gen(v, block_size):
    d = len(v)
    current_block = 0
    full_blocks = d // block_size
    has_tailing_block = ((d % block_size) != 0)
    n_blocks = full_blocks + has_tailing_block
    while current_block < n_blocks:
        yield v[current_block * block_size: min(d, (current_block + 1) * block_size)]
        current_block += 1
    
def quantize_single_block(v_block, p_norm=2):
    block_norm = np.linalg.norm(v_block, ord=p_norm)
    if (block_norm == 0):
        return v_block
    xi = np.random.uniform(size=len(v_block)) < (np.abs(v_block) / block_norm)
    return xi * np.sign(v_block)
        
def quantize(v, p_norm=2, block_size=1):
    block_gen_for_norms = block_gen(v, block_size)
    block_norms = np.array([np.linalg.norm(v_block, ord=p_norm) for v_block in block_gen_for_norms])
    block_gen_for_signs = block_gen(v, block_size)
    if s == 1:
        signs_quantized = np.concatenate([quantize_single_block(v_block, p_norm) for v_block in block_gen_for_signs])
        pos_neg = np.concatenate([signs_quantized > 0, signs_quantized < 0])
        return block_norms, np.packbits(pos_neg)

def decompress(block_norms, signs_compressed, d, p_norm=2, block_size=1):
    if s == 1:
        signs_pos_neg = np.unpackbits(signs_compressed)[:2 * d]
        decompressed_positive = signs_pos_neg[:d]
        decompressed_negative = signs_pos_neg[d:]
        signs = decompressed_positive * 1. - decompressed_negative * 1.
        sign_blocks = block_gen(signs, block_size)
        decompressed = np.concatenate([block_norms[i_block] * sign_block for i_block, sign_block in enumerate(sign_blocks)])
        assert len(decompressed) == d
        return decompressed