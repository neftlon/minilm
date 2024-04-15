import functools, math, typing
import jax, jax.numpy as jnp, jax.random as jr

class Embedding(typing.NamedTuple):
  vocab_size: int
  emb_dim: int

  def __call__(self, params, x):
    return params[x]

  def init(self, key):
    return jr.normal(key, (self.vocab_size, self.emb_dim)) / self.emb_dim

class Mlp(typing.NamedTuple):
  sizes: typing.Sequence[int]
  activation: str = "tanh"

  def __call__(self, params, x):
    activation = {"tanh": jax.nn.tanh, "relu": jax.nn.relu}[self.activation]
    
    for w, b in params[:-1]:
      x = jnp.dot(x, w) + b
      x = activation(x)
    w, b = params[-1]
    return jnp.dot(x, w) + b

  def init(self, key, scale=1e-2):
    params = []
    for n, m in zip(self.sizes, self.sizes[1:]):
      key, wkey, bkey = jr.split(key, 3)
      w, b = scale * jr.normal(wkey, (n,m)), scale * jr.normal(bkey, (m,))
      params.append((w, b))
    return params

  def __hash__(self):
    return hash(tuple(self.sizes))

class SelfAttention(typing.NamedTuple):
  d: int # model dimension
  dv: int
  dk: int

  def __call__(self, params, x, mask=False, return_scores=False):
    assert x.shape[-1] == self.d
    L = x.shape[-2] # sequence length
    
    # embed input
    Wq, Wk, Wv = params
    q, k, v = (jnp.dot(x, p) for p in (Wq, Wk, Wv))
    assert q.shape[-1] == self.dk and k.shape[-1] == self.dk
    assert v.shape[-1] == self.dv

    # compute attention scores
    alpha = jnp.einsum("...ij,...kj->...ik", q, k) / math.sqrt(self.dk)
    if mask:
      mask_ix = jnp.triu_indices(L, 1)
      alpha = alpha.at[..., *mask_ix].set(-jnp.inf)
    alpha = jax.nn.softmax(alpha)
    assert alpha.shape[-1] == alpha.shape[-2] == L

    # apply attention scores
    x = jnp.einsum("...ij,...jk->...ik", alpha, v)

    # construct output
    res = x
    if return_scores:
      res = res, alpha # append scores
    return res

  def init(self, key, scale=1e-2):
    d, dk, dv = self.d, self.dk, self.dv
    shapes = ((d, dk), (d, dk), (d, dv))
    keys = jr.split(key, len(shapes))
    return tuple(scale * jr.normal(k, s) for k, s in zip(keys, shapes))

class MhSelfAttention(typing.NamedTuple):
  num_heads: int
  head_template: SelfAttention

  def __call__(self, params, x, mask=False, return_scores=False):
    d, dv = self.head_template.d, self.head_template.dv
    assert x.shape[-1] == d
    L = x.shape[-2]
    heads, Wo = params

    # call heads in parallel
    head_fn = functools.partial(self.head_template, mask=mask, return_scores=True)
    x, alphas = jax.vmap(head_fn, (0, None), -1)(heads, x) # (...,L,dv,h), (...,L,L,h)
    assert x.shape[-3:] == (L, dv, self.num_heads)
    assert alphas.shape[-3:] == (L, L, self.num_heads)

    # flatten heads and project back into embedding space
    x = x.reshape(*x.shape[:-2], dv * self.num_heads) # concatenate last two dimensions
    x = jnp.dot(x, Wo) # (...,L,d)
    assert x.shape[-2:] == (L, d)

    # construct result
    res = x
    if return_scores:
      # move head axis before LxL dimensions
      res = res, alphas.transpose(*range(alphas.ndim - 3), -1, -3, -2)
    return res

  def init(self, key, scale=1e-2):
    # initialize output mapping
    dv, d = self.head_template.dv, self.head_template.d
    key, wokey = jr.split(key)
    Wo = scale * jr.normal(wokey, (self.num_heads * dv, d))
    
    # initialize heads
    keys = jr.split(key, self.num_heads)
    heads = jax.vmap(self.head_template.init)(keys)
    
    return heads, Wo
  
class LayerNorm(typing.NamedTuple):
  norm_shape: typing.Sequence[int]
  eps: float = 1e-5

  def __call__(self, params, x):
    gam, beta = params
    axs = range(-len(self.norm_shape), 0) # axes to average over
    mean, var = jnp.mean(x, axis=axs, keepdims=True), jnp.var(x, axis=axs, keepdims=True)
    return (x - mean) / jnp.sqrt(var + self.eps) * gam + beta

  def init(self, key):
    return jnp.ones(self.norm_shape), jnp.zeros(self.norm_shape)

  def __hash__(self):
    return hash((tuple(self.norm_shape), self.eps))

class MhSaDecBlock(typing.NamedTuple):
  ln1: LayerNorm
  attn: MhSelfAttention
  ff: Mlp
  ln2: LayerNorm

  def __call__(self, params, x, return_scores=False):
    # pre-apply parameters to layers
    ln1, attn, ff, ln2 = (functools.partial(l, p) for l, p in zip(self, params))

    # apply first layer norm and then attention
    x = ln1(x)
    y, alphas = attn(x, mask=True, return_scores=True)
    x = x + y

    # apply MLP and then second layer norm
    x = ln2(x + ff(x))
    
    return (x, alphas) if return_scores else x
    
  def init(self, key):
    keys = jr.split(key, len(self))
    return tuple(l.init(k) for l, k in zip(self, keys))

  @classmethod
  def from_spec(cls, emb_dim: int, num_heads: int, hidden_dim: int):
    return cls(
      ln1=LayerNorm((emb_dim,)),
      attn=MhSelfAttention(num_heads, SelfAttention(emb_dim, emb_dim // num_heads, emb_dim // num_heads)),
      ff=Mlp([emb_dim, hidden_dim, emb_dim], activation="relu"),
      ln2=LayerNorm((emb_dim,)),
    )

class Miniformer(typing.NamedTuple):
  ce: Embedding # char embedding
  pe: Embedding # positional embedding
  num_blocks: int
  block_template: MhSaDecBlock

  def __call__(self, params, x, return_scores=False):
    L = x.shape[-1]
    # pre-apply parameters to layers
    ce, pe = (functools.partial(l, p) for l, p in zip((self.ce, self.pe), params))
    ce_params = params[0]

    # run input through model
    x = ce(x) + pe(jnp.arange(L)) # positional embedding will be cast to batch
    alphas = []
    for block_params in params[-1]:
      x, block_alphas = self.block_template(block_params, x, return_scores=True)
      alphas.append(block_alphas)
    alphas = jnp.stack(alphas, axis=-4) # (..., num_blocks, h, L, L)
    x = jnp.dot(x, ce_params.T) # deembed output

    return (x, alphas) if return_scores else x

  def init(self, key):
    keys = jr.split(key, 3)
    block_params = [self.block_template.init(k) for k in jr.split(keys[2], self.num_blocks)]
    assert len(block_params) == self.num_blocks
    return self.ce.init(keys[0]), self.pe.init(keys[1]), block_params

  @classmethod
  def from_spec(cls, num_blocks: int, vocab_size: int, emb_dim: int, num_heads: int, hidden_dim: int):
    return cls(
      ce=Embedding(vocab_size, emb_dim),
      pe=Embedding(vocab_size, emb_dim),
      num_blocks=num_blocks,
      block_template=MhSaDecBlock.from_spec(emb_dim, num_heads, hidden_dim),
    )
