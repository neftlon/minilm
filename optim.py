import typing
import jax, jax.numpy as jnp

class Gd(typing.NamedTuple):
  alpha: float = 1e-2

  def step(self, state, params, grads):
    assert state == ()
    return state, jax.tree.map(lambda p, g: p - self.alpha * g, params, grads)
  
  def init(self, _params):
    return () # no state

class Adam(typing.NamedTuple):
  alpha: float = 1e-3
  betas: tuple[float, float] = (0.9, 0.999)
  eps: float = 1e-5

  def step(self, state, params, grads):
    m, v, t = state
    t = t + 1
    m = jax.tree.map(lambda m, g: self.betas[0] * m + (1 - self.betas[0]) * g, m, grads)
    v = jax.tree.map(lambda v, g: self.betas[1] * v + (1 - self.betas[1]) * g ** 2, v, grads)
    mh = jax.tree.map(lambda m: m / (1 - self.betas[0] ** t), m)
    vh = jax.tree.map(lambda v: v / (1 - self.betas[1] ** t), v)
    return (m, v, t), jax.tree.map(lambda p, mh, vh: p - self.alpha * mh / (jnp.sqrt(vh) + self.eps), params, mh, vh)
  
  def init(self, params):
    m = jax.tree.map(lambda p: jnp.zeros_like(p), params) # 1st moment
    v = jax.tree.map(lambda p: jnp.zeros_like(p), params) # 2nd moment
    return m, v, jnp.array(0)

if __name__ == "__main__":
  opt = Adam(alpha=.25)
  f = lambda x: x ** 2
  df = jax.grad(f)
  x0 = 2.4
  opt_state = opt.init(x0)
  x = x0
  for _ in range(50):
    grads = df(x)
    opt_state, x = opt.step(opt_state, x, grads)
    print(x)
