from __future__ import annotations

from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# # Muon optimizer aka MomentUm Orthogonalized


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps # normalise to consistent magnitude, eps prevents division by zero
    transposed = G.size(0) > G.size(1) # check if rows > cols
    if transposed:
        X = X.T # flip so iteration always works on smaller dimension (efficiency) e.g m * n if n < m use X.T because n * n will be smaller than m * m
    for _ in range(steps):
        A = X @ X.T # compute gram matrix
        B = b * A + c * A @ A  # polynomial terms
        X = a * X + B @ X # iterate toward U @ V.T
    return X.T if transposed else X # flip back to original shape if transposed

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
#       gpu setup
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
#              dictionary call for values
#           params is a list of 2d matrices
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
#                               numel stands for number of elements
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr=0
            for i, p in enumerate(params):
                #checks which gpu should process this param (matrix)
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g) # buf = momentum * buf + g (momentum is a hyperparamter to tell how much of previous momentum to keep)
                    if nesterov:
                        #nesterov essentially calculates momentum using future position
                        g = g.add(buf, alpha=momentum) # gradient = gradient + (momentum * buffer)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementation
                    g *= max(1, g.size(0)/g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            
            if distributed:
                #combine the update flat from each gpu
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            #reset the curr to zero
            curr = 0
            #for each 2d matrix
            #each p in params consist of 2d matrix weights and 2d gradient (based on nn.parameter and Tensor.grad)
            for p in params:
                #set the gradient as the section in the update flats, shaped as p, and matching the data type
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                #weight = weight - lr * gradient
                p.add_(g, alpha=-lr)
                #update curr to end of latest gradient
                curr += p.numel()
        
        return loss



# # Works by asking what is the most optimum update based on the structure of the weight matrix

# # works by finding the most efficient update direction
# # by removing redundancy across the rows and columns of the gradient matrix

# # the key is orthogonalisation

# # Similar to adam it works by improving the gradient through Orthogonalization.

# # Mathematically the cleanest version of a matrix with orthogonal structure comes from its SVD (Singular Value Decomposition)

# # @ means matrix multiplication

# # .T means transposed (swap axis)

# # U = eigenvectors of G @ G.T

# # V = eigenvectors of G.T @ G

# # G = U @ S @ V.T

# # with no magnitude S

# # G = U @ V.T

# # SVD is expensive especially with large matrices so we use Newton-Schulz which give an approximate

# We skip SVD entirely since we use Newton-Schulz

# # X = G / G.norm()          # normalise to start
# # for _ in range(5):        # iterate 5 times
# #     A = X @ X.T
# #     X = a*X + (b*A + c*A@A) @ X    # polynomial update
# # ```

# # Each iteration makes `X` closer to `U @ V.T`. After just 5 iterations it's close enough to be useful — much faster than full SVD.

# # The coefficients `a, b, c = (3.4445, -4.7750, 2.0315)` are carefully chosen so the polynomial converges as fast as possible in those 5 steps rather than slowly creeping toward the answer.

# # So the full Muon update is:
# # ```
# # gradient → normalise → 5x Newton-Schulz iterations → approximately U @ V.T → step

# # Muon is applied to structured 2D weight matrices (attention, MLP)
# # where row/column relationships exist and redundancy is meaningful
# # Adam is used for embeddings where adaptive scaling matters more

# A = G @ G.T = [[10, 5],
#                [ 5, 5]]
# ```

# **What we're looking for**

# We want vectors that only get stretched by A, never rotated:
# ```
# A @ v = λ * v
# ```

# Where `v` is the eigenvector and `λ` (lambda) is the eigenvalue — the scaling factor. This equation is saying "multiplying A by v gives back the same v, just scaled by λ."

# **How you solve it**

# Rearrange the equation:
# ```
# A @ v - λ * v = 0
# (A - λI) @ v  = 0      ← I is the identity matrix
# ```

# For this to have a solution, the determinant of `(A - λI)` must equal zero. That gives you the characteristic equation:
# ```
# det([[10-λ,  5  ],
#      [ 5,   5-λ]]) = 0

# (10-λ)(5-λ) - (5*5) = 0
# 50 - 15λ + λ² - 25 = 0
# λ² - 15λ + 25       = 0
# ```

# Solving this quadratic gives you the eigenvalues:
# ```
# λ1 = 12.808
# λ2 = 2.192
# ```

# **Then plug each eigenvalue back in to get the eigenvectors**

# For λ1 = 12.808:
# ```
# (A - 12.808 * I) @ v = 0

# [[-2.808,  5   ],   @   [v1]   =   [0]
#  [ 5,     -7.808]]      [v2]       [0]
# ```

# Solving gives `v = [0.851, 0.526]` — that's eigenvector 1, which becomes a column of U.

# For λ2 = 2.192:
# ```
# solving gives v = [0.526, -0.851]  — eigenvector 2, second column of U
# ```

# So:
# ```
# U = [[0.851,  0.526],
#      [0.526, -0.851]]


# In practice you never do this by hand — you just call:
# U, S, Vt = torch.linalg.svd(G)