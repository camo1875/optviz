"""보고서의 결론을 '한 번의 실행'이 아니라 여러 조건에서 확인해보는 실험들.

보고서는 시작점 (-0.9, 0.5), η = 0.01 한 가지 조건에서 Adam(β1=0.9)이 극솟값에 갇히고
β1=0.99로 바꾸면 최솟값을 찾는다는 결론을 냈다. 아래 실험으로 그 결론이 시작점과 하이퍼파라미터에
얼마나 의존하는지 볼 수 있다. 옵티마이저가 벡터화되어 있어 수만 개의 조건을 한 번에 돌린다.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import optimizers as O
from .visualize import draw_contour

# 도착 결과별 색: 수렴 못함, 최솟값, 극솟값 #1, #2, ...
RESULT_COLORS = ["#ced4da", "#40c057", "#fa5252", "#fd7e14", "#7950f2"]


def _result_cmap(fn):
    n = len(fn.minima) + 1
    return ListedColormap(RESULT_COLORS[:n]), n


def _result_labels(fn):
    return ["not converged", "global min"] + [f"local min #{i}" for i in range(1, len(fn.minima))
                                              if not fn.is_global(i)]


def _final_result(fn, path):
    """경로 끝이 어느 극소점인지 (+1 해서 0이 '수렴 못함'이 되게)"""
    last = np.where(np.isfinite(path[-1]), path[-1], 1e9)
    return fn.which_minimum(last) + 1


def beta1_paths(fn, betas=(0.8, 0.9, 0.95, 0.99), lr=None, steps=None, start=None):
    """보고서 37쪽 실험: Adam의 β1만 바꿔가며 경로를 비교한다."""
    steps = steps or fn.steps
    start = fn.start if start is None else start
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_contour(ax, fn)
    cmap = plt.get_cmap("plasma")
    print(f"Adam β1 sweep  (start={tuple(start)}, steps={steps})")
    for i, b1 in enumerate(betas):
        opt = O.make("Adam", fn, lr=lr, beta1=b1)
        p = O.run(opt, fn, start, steps)
        ax.plot(p[:, 0], p[:, 1], color=cmap(i / max(len(betas) - 1, 1) * 0.85), lw=2, label=f"β1 = {b1}")
        print(f"  β1={b1:<5}  final=({p[-1, 0]:.4f}, {p[-1, 1]:.4f})  "
              f"→ {_result_labels(fn)[int(_final_result(fn, p))]}")
    ax.plot(*start, "s", color="black", ms=6, label="start")
    ax.set_title(f"Adam: effect of β1  (η={opt.lr}, {steps} steps)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def basin_map(fn, names=O.REPORT_OPTIMIZERS, res=121, steps=None):
    """모든 시작점에 대해, 각 옵티마이저가 최종적으로 어느 극소점에 도착하는지 색칠한다."""
    steps = steps or fn.steps
    xs, ys = np.linspace(*fn.xlim, res), np.linspace(*fn.ylim, res)
    starts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    cmap, n = _result_cmap(fn)

    cols = min(len(names), 3)
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    print(f"basin map: {res}x{res} start points, {steps} steps — share of start points reaching the global min")
    for ax, name in zip(axes.flat, names):
        p = O.run(O.make(name, fn), fn, starts, steps)
        result = _final_result(fn, p).reshape(res, res)
        ax.imshow(result, origin="lower", extent=(*fn.xlim, *fn.ylim), cmap=cmap, vmin=-0.5, vmax=n - 0.5,
                  aspect="auto", interpolation="nearest")
        for mx, my, _ in fn.minima:
            ax.plot(mx, my, "k+", ms=10)
        ax.plot(*fn.start, "s", mfc="white", mec="black", ms=6)
        share = (result == 1).mean()
        ax.set_title(f"{name}  ({share:.0%} → global min)")
        print(f"  {name:9s} {share:6.1%}")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    _legend(fig, fn)
    fig.suptitle(f"Where does each optimizer end up?  ({steps} steps, square = report's start point)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def hparam_map(fn, name="Adam", param="beta1", values=None, lrs=None, steps=None, start=None):
    """학습률 × 하이퍼파라미터(기본: Adam의 β1) 조합마다 도착 결과를 색칠한다.
    보고서의 'β1 = 0.9 → 0.99' 결과가 학습률이 달라져도 성립하는지 볼 수 있다."""
    steps = steps or fn.steps
    start = fn.start if start is None else start
    # β1 = 0.9, 0.99, 0.999 처럼 1에 가까운 값이 중요하므로 1-β1을 로그 간격으로 잡는다
    values = 1 - np.geomspace(1, 2e-3, 120) if values is None else np.asarray(values)
    lrs = np.geomspace(1e-3, 0.3, 120) if lrs is None else np.asarray(lrs)
    V, L = np.meshgrid(values, lrs)
    opt = O.make(name, fn, lr=L.reshape(-1, 1), **{param: V.reshape(-1, 1)})
    p = O.run(opt, fn, np.broadcast_to(np.asarray(start, float), (V.size, 2)), steps)
    result = _final_result(fn, p).reshape(V.shape)

    cmap, n = _result_cmap(fn)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(1 - V, L, result, cmap=cmap, vmin=-0.5, vmax=n - 0.5, shading="nearest")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ticks = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995]
    ax.set_xticks([1 - t for t in ticks], [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(f"{param}  (log scale on 1 - {param})")
    ax.set_ylabel("learning rate η")
    default_lr = fn.adaptive_lr if O.OPTIMIZERS[name].adaptive else fn.lr
    ax.plot([0.1, 0.01], [default_lr] * 2, "s", mfc="white", mec="black", ms=7)
    ax.set_title(f"{name} from start {tuple(start)}, {steps} steps  (squares: report's settings)")
    _legend(fig, fn)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def _legend(fig, fn):
    labels = _result_labels(fn)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RESULT_COLORS[i]) for i in range(len(labels))]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
