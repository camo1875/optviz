"""Experiments that check the report's conclusions over many settings instead of a single run.

The report used one setting (start (-0.9, 0.5), η = 0.01) and concluded that Adam (β1 = 0.9) gets stuck in
the local minimum and finds the global minimum with β1 = 0.99. These experiments show how much that depends
on the start point and the hyperparameters. The optimizers are vectorized, so tens of thousands of
settings run at once.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import optimizers as O
from .visualize import draw_contour

# colors per outcome: not converged, global min, local min #1, #2, ...
RESULT_COLORS = ["#ced4da", "#40c057", "#fa5252", "#fd7e14", "#7950f2"]


def _result_cmap(fn):
    n = len(fn.minima) + 1
    return ListedColormap(RESULT_COLORS[:n]), n


def _result_labels(fn):
    return ["not converged", "global min"] + [f"local min #{i}" for i in range(1, len(fn.minima))
                                              if not fn.is_global(i)]


def _final_result(fn, path):
    """Which minimum the path ends at, shifted by +1 so that 0 means 'not converged'."""
    last = np.where(np.isfinite(path[-1]), path[-1], 1e9)
    return fn.which_minimum(last) + 1


def beta1_paths(fn, betas=(0.8, 0.9, 0.95, 0.99), lr=None, steps=None, start=None):
    """Report p.37 experiment: compare Adam paths while changing only β1."""
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
    """For every start point, color the minimum each optimizer finally reaches."""
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
    """Color the outcome for every (learning rate, hyperparameter) pair (default: Adam's β1).
    Shows whether the report's 'β1 = 0.9 → 0.99' result still holds at other learning rates."""
    steps = steps or fn.steps
    start = fn.start if start is None else start
    # values close to 1 (0.9, 0.99, 0.999) matter most, so space 1-β1 logarithmically
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
