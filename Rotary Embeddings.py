import torch

class Rotary(nn.Module):
    #Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim:int, base:float = 10000.0):
        super().__init__()
        #formula for calculating base frequency values
        inv_freq = 1.0 / (base ** (torch.arrange(0, dim, 2, dtype=torch.float32)/dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    
    def forward(self, seq_len:int, device: torch.device, dtype: torch.dtype)-> tuple[Tensor, Tensor]: #return tuple
        if(#base checks
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) # setup the position indices
            freqs = torch.outer(t, self.inv_freq.to(device)) #insert values into the matrix using the inv_freq formula and store it on the same device
            self._cos_cached = freqs.cos()[None,None,:,:] #applies cos to the freqs matrix None is used to broadcast (automate expansion of dimension)
            self._sin_cached = freqs.sin()[None,None,:,:] #applies sin to the freqs matrix None is used to broadcast (automate expansion of dimension)
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype) #return the cos and sin applied freqs matrices
    

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor: #return the position integrated matrix
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[...,half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

rope_base: float
#example class rotary
self.rotary = Rotary(self.head_dim, base=rope_base)
#example initialize cos and sin
cos, sin = self.rotary(seqlen, x.device, q.dtype)
#example applying rope
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)

#cos and sin are x and y axis distance/ratio/percentage from a point