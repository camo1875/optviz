"""시각화에 쓰는 2변수 손실 함수들.

보고서와 같이 sympy로 식을 적고 편미분을 자동으로 구하지만, 실제 계산은 lambdify한 numpy 함수로 한다.
(원본은 매 스텝 subs로 기호 계산을 해서 느렸다.)
임계점도 그래디언트 = 0 을 풀고 헤시안의 고윳값으로 극소/안장점을 자동 분류한다.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

X, Y = sp.symbols("x y", real=True)
R = sp.Rational


@dataclass
class LossFunction:
    key: str
    title: str
    expr: sp.Expr
    xlim: tuple
    ylim: tuple
    start: tuple
    lr: float  # GD 계열 기본 학습률
    adaptive_lr: float = None  # Adagrad/RMSProp/Adam 계열 기본 학습률 (None이면 lr과 같음)
    steps: int = 500
    minima: list = field(init=False)  # [(x, y, f값)]  f값 오름차순
    saddles: list = field(init=False)
    maxima: list = field(init=False)

    def __post_init__(self):
        if self.adaptive_lr is None:
            self.adaptive_lr = self.lr
        gx, gy = sp.diff(self.expr, X), sp.diff(self.expr, Y)
        self._f = sp.lambdify((X, Y), self.expr, "numpy")
        self._gx = sp.lambdify((X, Y), gx, "numpy")
        self._gy = sp.lambdify((X, Y), gy, "numpy")
        self._classify_critical_points(gx, gy)

    def _classify_critical_points(self, gx, gy):
        hessian = sp.lambdify((X, Y), sp.hessian(self.expr, (X, Y)), "numpy")
        self.minima, self.saddles, self.maxima = [], [], []
        for sol in sp.solve([gx, gy], [X, Y], dict=True):
            px, py = complex(sol[X].evalf()), complex(sol[Y].evalf())
            if abs(px.imag) > 1e-9 or abs(py.imag) > 1e-9:
                continue
            p = (px.real, py.real)
            eig = np.linalg.eigvalsh(np.array(hessian(*p), dtype=float))
            target = self.minima if eig.min() > 0 else self.maxima if eig.max() < 0 else self.saddles
            target.append((*p, float(self._f(*p))))
        self.minima.sort(key=lambda m: m[2])

    @property
    def f_min(self):
        return self.minima[0][2]

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        x, y = theta[..., 0], theta[..., 1]
        return np.broadcast_to(self._f(x, y), x.shape).astype(float)

    def grad(self, theta):
        theta = np.asarray(theta, dtype=float)
        x, y = theta[..., 0], theta[..., 1]
        # 미분 결과가 상수이면 lambdify가 스칼라를 돌려주므로 모양을 맞춰준다
        return np.stack([np.broadcast_to(self._gx(x, y), x.shape),
                         np.broadcast_to(self._gy(x, y), x.shape)], axis=-1).astype(float)

    def which_minimum(self, theta, tol=0.05):
        """theta가 어느 극소점에 수렴했는지. 0이 최솟값, 가장 가까운 극소점과 tol 이상 떨어져 있으면 -1
        (함숫값이 최솟값과 같은 극소점이 여러 개면 모두 0)"""
        theta = np.asarray(theta, dtype=float)
        mins = np.array([m[:2] for m in self.minima])
        dist = np.linalg.norm(theta[..., None, :] - mins, axis=-1)
        idx = dist.argmin(axis=-1)
        close = np.take_along_axis(dist, idx[..., None], -1)[..., 0] < tol
        return np.where(close, np.where(self.is_global(idx), 0, idx), -1)

    def is_global(self, i):
        return np.abs(np.array([m[2] for m in self.minima])[i] - self.f_min) < 1e-9


FUNCTIONS = {f.key: f for f in [
    LossFunction(
        "report", "f(x,y) = x^4 - x^3/2 - 3x^2/4 - x/8 + y^2 - 1/2  (report 4.6)",
        X**4 - R(1, 2)*X**3 - R(3, 4)*X**2 - R(1, 8)*X + Y**2 - R(1, 2),
        xlim=(-1.4, 1.4), ylim=(-1.0, 1.0), start=(-0.9, 0.5), lr=0.01,
    ),
    LossFunction(
        "bowl", "f(x,y) = x^2/20 + y^2  (ill-conditioned bowl)",
        X**2/20 + Y**2,
        xlim=(-5, 5), ylim=(-2, 2), start=(-4.5, 1.5), lr=0.1, steps=300,
    ),
    LossFunction(
        "saddle", "f(x,y) = x^2 + y^4/4 - y^2/2  (saddle point)",
        X**2 + Y**4/4 - Y**2/2,
        xlim=(-2, 2), ylim=(-1.6, 1.6), start=(1.5, 1e-3), lr=0.05, steps=300,
    ),
    LossFunction(
        "rosenbrock", "f(x,y) = (1-x)^2 + 100(y-x^2)^2  (Rosenbrock)",
        (1 - X)**2 + 100*(Y - X**2)**2,
        xlim=(-2, 2), ylim=(-1, 3), start=(-1.5, 2.0), lr=1e-3, adaptive_lr=0.02, steps=2000,
    ),
]}
