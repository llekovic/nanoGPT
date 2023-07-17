"""Custom attention implementations"""

import torch
import math
import torch.nn.functional as F
import gpytorch


"""
Ideal implementaion would compute dropout and final matmul (as outlined below), but I can't pass <self.attn_dropout> through kernel_attention_...

This would be better because I could make <dropout=self.dropout if self.training else 0> (Right now, dropout is included even when outside training...)

    # Ideal kernel implementation
    scores = kernel(q,k).cuda().evaluate()      #  (B x num heads x T x T)
    scores = scores / math.sqrt(k.size(-1))
    scores = scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('0'))
    probs = scores / scores.sum(dim=2, keepdim=True)    # Normalize the attention scores to probabilities (no softmax)
    probs = self.attn_dropout(probs)
    y = probs @ v

NOTE: Should be able to use "with settings.lazily_evaluate_kernels(False):", but I can't get it to work (instead I am just using .evaluate())
"""
def k_attention(q, k, bias, profiler):
    # Implementation of k @ k attention
    _, _, T, _ = k.size()
    att = (k @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
    y = F.softmax(att, dim=-1)
    return y

#working kernel(K) attention
def kernel_attention_k(k, bias, kernel, profiler):
    _, _, T, hs = k.size()
    k = k / math.sqrt(k.size(-1))
    # Compute RBF kernel to get attention scores
    scores = kernel(k).evaluate()        #  (B x num heads x T x T)
    scores = scores.masked_fill(bias[:,:,:T,:T], float('0'))
    y = scores / (scores.sum(dim=-1, keepdim=True) + 1e-4)    # Normalize the attention scores to probabilities (no softmax)
    return y

def kernel_attention_qk(q, k, bias, kernel, profiler):
    with profiler("divide by sqrt k.size(-1)"):
        _, _, T, hs = k.size()
        q = q / math.sqrt(k.size(-1))
        k = k / math.sqrt(k.size(-1))
    # Compute kernel to get attention scores
    with profiler("kernel calculation"):
        scores = kernel(q,k).evaluate()        #  (B x num heads x T x T)
    with profiler("masking"):
        scores = scores.masked_fill(bias[:,:,:T,:T], float('0'))
    with profiler("normalizing"):
        y = scores / (scores.sum(dim=-1, keepdim=True) + 1e-4)   # Normalize the attention scores to probabilities (no softmax)
    return y






