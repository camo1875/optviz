"""The optimizers from chapter 4 of the report.

Every operation is element-wise (Hadamard), so theta can be a single point of shape (2,) or many points
of shape (N, 2). Hyperparameters can also be (N, 1) arrays to run N settings at once (used in experiments.py).

Notation follows the report:  g = ∇f(Θ_{n-1}),  η = learning rate (lr)
"""
import numpy as np


class Optimizer:
    name = "base"
    adaptive = False  # adaptive learning rate (Adagrad family)? Used to pick the default learning rate

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.reset()

    def reset(self):
        self.t = 0

    def step(self, theta, grad_fn):
        """Return Θ_n = Θ_{n-1} + Δ."""
        self.t += 1
        return theta + self.delta(theta, grad_fn)

    def delta(self, theta, grad_fn):
        raise NotImplementedError

    def _zeros(self, theta):
        return np.zeros(np.broadcast_shapes(np.shape(theta), np.shape(self.lr)))


class GD(Optimizer):
    """Θ_n = Θ_{n-1} - η∇f(Θ_{n-1})"""
    name = "GD"

    def delta(self, theta, grad_fn):
        return -self.lr * grad_fn(theta)


class Momentum(Optimizer):
    """v_n = γ v_{n-1} - η∇f(Θ_{n-1}),  Θ_n = Θ_{n-1} + v_n"""
    name = "Momentum"

    def __init__(self, lr=0.01, gamma=0.9, **kw):
        self.gamma = gamma
        super().__init__(lr, **kw)

    def reset(self):
        super().reset()
        self.v = None

    def delta(self, theta, grad_fn):
        if self.v is None:
            self.v = self._zeros(theta)
        self.v = self.gamma * self.v - self.lr * grad_fn(theta)
        return self.v


class NAG(Momentum):
    """v_n = γ v_{n-1} - η∇f(Θ_{n-1} + γ v_{n-1}),  Θ_n = Θ_{n-1} + v_n"""
    name = "NAG"

    def delta(self, theta, grad_fn):
        if self.v is None:
            self.v = self._zeros(theta)
        self.v = self.gamma * self.v - self.lr * grad_fn(theta + self.gamma * self.v)
        return self.v


class Adagrad(Optimizer):
    """G_n = G_{n-1} + g⊙g,  Θ_n = Θ_{n-1} - η/√(G_n+ε) ⊙ g"""
    name = "Adagrad"
    adaptive = True

    def reset(self):
        super().reset()
        self.G = None

    def delta(self, theta, grad_fn):
        g = grad_fn(theta)
        if self.G is None:
            self.G = self._zeros(theta)
        self.G = self.G + g * g
        return -self.lr * g / np.sqrt(self.G + self.eps)


class RMSProp(Adagrad):
    """G_n = γ G_{n-1} + (1-γ) g⊙g,  Θ_n = Θ_{n-1} - η/√(G_n+ε) ⊙ g"""
    name = "RMSProp"

    def __init__(self, lr=0.01, gamma=0.9, **kw):
        self.gamma = gamma
        super().__init__(lr, **kw)

    def delta(self, theta, grad_fn):
        g = grad_fn(theta)
        if self.G is None:
            self.G = self._zeros(theta)
        self.G = self.gamma * self.G + (1 - self.gamma) * g * g
        return -self.lr * g / np.sqrt(self.G + self.eps)


class Adam(Optimizer):
    """M_n = β1 M_{n-1} + (1-β1) g,  V_n = β2 V_{n-1} + (1-β2) g⊙g
    M̂_n = M_n/(1-β1^n),  V̂_n = V_n/(1-β2^n),  Θ_n = Θ_{n-1} - η/√(V̂_n+ε) ⊙ M̂_n

    ε sits inside the square root, as in the report. (PyTorch and others use √V̂_n + ε; the difference is negligible.)
    """
    name = "Adam"
    adaptive = True

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, **kw):
        self.beta1, self.beta2 = beta1, beta2
        super().__init__(lr, **kw)

    def reset(self):
        super().reset()
        self.M = self.V = None

    def _moments(self, theta, grad_fn):
        g = grad_fn(theta)
        if self.M is None:
            self.M, self.V = self._zeros(theta), self._zeros(theta)
        self.M = self.beta1 * self.M + (1 - self.beta1) * g
        self.V = self.beta2 * self.V + (1 - self.beta2) * g * g
        return g, self.M / (1 - self.beta1 ** self.t), self.V / (1 - self.beta2 ** self.t)

    def delta(self, theta, grad_fn):
        _, m_hat, v_hat = self._moments(theta, grad_fn)
        return -self.lr * m_hat / np.sqrt(v_hat + self.eps)


class Nadam(Adam):
    """Adam + NAG (Dozat, 2016). Not part of the report.
    Like Momentum → NAG, it uses the look-ahead momentum β1 M̂_n + (1-β1) g/(1-β1^n) instead of M̂_n.
    """
    name = "Nadam"

    def delta(self, theta, grad_fn):
        g, m_hat, v_hat = self._moments(theta, grad_fn)
        lookahead = self.beta1 * m_hat + (1 - self.beta1) * g / (1 - self.beta1 ** self.t)
        return -self.lr * lookahead / np.sqrt(v_hat + self.eps)


OPTIMIZERS = {cls.name: cls for cls in [GD, Momentum, NAG, Adagrad, RMSProp, Adam, Nadam]}
REPORT_OPTIMIZERS = ["GD", "Momentum", "NAG", "Adagrad", "RMSProp", "Adam"]
COLORS = {"GD": "#e8590c", "Momentum": "#2f9e44", "NAG": "#1971c2", "Adagrad": "#9c36b5",
          "RMSProp": "#c2255c", "Adam": "#212529", "Nadam": "#0c8599"}


def make(name, fn=None, lr=None, **kw):
    """Create an optimizer by name. Without lr, use the default learning rate of loss function fn."""
    cls = OPTIMIZERS[name]
    if lr is None:
        lr = fn.adaptive_lr if cls.adaptive else fn.lr
    return cls(lr=lr, **kw)


def run(optimizer, fn, start, steps):
    """Run `steps` updates from `start` and return the path as an array of shape (steps+1, ..., 2).
    If a run diverges (inf/nan or very large values), the rest of its path is left as nan."""
    optimizer.reset()
    theta = np.array(start, dtype=float)
    path = np.full((steps + 1,) + theta.shape, np.nan)
    path[0] = theta
    alive = np.ones(theta.shape[:-1], dtype=bool)
    with np.errstate(all="ignore"):
        for n in range(1, steps + 1):
            theta = optimizer.step(theta, fn.grad)
            alive &= np.isfinite(theta).all(axis=-1) & (np.abs(theta).max(axis=-1) < 1e6)
            if not alive.any():
                break
            path[n] = np.where(alive[..., None], theta, np.nan)
    return path
