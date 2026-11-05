"""Optimizer path plots: 3D surface + top-view contours + loss gap in one figure.

Changes from the original (report 4.6)
- Only the region the paths actually visit is drawn, so the paths are visible
  (the original used [-5, 5], where z grows to about 600)
- Top-view contour map with the global minimum / local minima / saddle points marked
- f(Θ_n) - f_min per step on a log scale, to compare convergence speed
- FuncAnimation instead of plt.ion + time.sleep, so the animation can be saved as GIF/MP4
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from .optimizers import COLORS


def grid(fn, res=120):
    xs = np.linspace(*fn.xlim, res)
    ys = np.linspace(*fn.ylim, res)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, fn(np.stack([gx, gy], axis=-1))


def draw_contour(ax, fn, res=200):
    """Draw contours and critical points. Also used by experiments.py."""
    gx, gy, z = grid(fn, res)
    # pick levels by quantile so contours are denser near the bottom
    levels = np.unique(np.quantile(z, np.linspace(0, 1, 30) ** 2))
    ax.contourf(gx, gy, z, levels=levels, cmap="Greys_r", alpha=0.35)
    ax.contour(gx, gy, z, levels=levels, colors="grey", linewidths=0.4, alpha=0.6)
    labeled = set()
    for i, (mx, my, _) in enumerate(fn.minima):
        g = bool(fn.is_global(i))
        label = "global min" if g else "local min"
        ax.plot(mx, my, marker="*" if g else "o", ms=14 if g else 8, mfc="gold" if g else "white",
                mec="black", zorder=5, label=None if label in labeled else label)
        labeled.add(label)
    for j, (sx, sy, _) in enumerate(fn.saddles):
        ax.plot(sx, sy, marker="X", ms=9, mfc="white", mec="black", zorder=5, label="saddle" if j == 0 else None)
    ax.set_xlim(fn.xlim)
    ax.set_ylim(fn.ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


class Scene:
    """Figure that draws every path up to step n. Calling update(n) with increasing n animates it."""

    def __init__(self, fn, paths, title=None):
        self.fn, self.paths = fn, paths
        self.steps = len(next(iter(paths.values()))) - 1
        self.losses = {k: fn(p) - fn.f_min for k, p in paths.items()}

        self.fig = plt.figure(figsize=(17, 5.6))
        self.fig.suptitle(title or fn.title)
        ax3 = self.fig.add_subplot(1, 3, 1, projection="3d")
        axc = self.fig.add_subplot(1, 3, 2)
        axl = self.fig.add_subplot(1, 3, 3)

        gx, gy, z = grid(self.fn, 60)
        ax3.plot_surface(gx, gy, z, cmap="viridis", alpha=0.35, linewidth=0, antialiased=True)
        ax3.set_xlim(fn.xlim)
        ax3.set_ylim(fn.ylim)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.view_init(elev=40, azim=-60)
        draw_contour(axc, fn)
        axc.set_title("top view")

        floor = min(np.nanmin(l[np.isfinite(l)]) for l in self.losses.values() if np.isfinite(l).any())
        axl.set_yscale("log")
        axl.set_xlim(0, self.steps)
        axl.set_ylim(max(floor, 1e-12) / 3, max(np.nanmax(l) for l in self.losses.values()) * 3)
        axl.set_xlabel("step n")
        axl.set_ylabel("f(Θ_n) - f_min")
        axl.set_title("loss gap (log scale)")
        axl.grid(alpha=0.3, which="both")

        self.artists = {}
        for name, path in paths.items():
            c = COLORS.get(name)
            self.artists[name] = (
                ax3.plot([], [], [], color=c, lw=2.5)[0],
                axc.plot([], [], color=c, lw=2, label=name)[0],
                axc.plot([], [], "o", color=c, ms=6, mec="white")[0],
                axl.plot([], [], color=c, lw=1.8)[0],
            )
        axc.plot(*paths[next(iter(paths))][0], "s", color="black", ms=6, label="start")
        axc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        self.fig.tight_layout()

    def update(self, n):
        for name, (line3, line2, head, lossline) in self.artists.items():
            p = self.paths[name][: n + 1]
            line3.set_data(p[:, 0], p[:, 1])
            line3.set_3d_properties(self.fn(p))
            line2.set_data(p[:, 0], p[:, 1])
            head.set_data(p[-1:, 0], p[-1:, 1])
            lossline.set_data(np.arange(n + 1), np.maximum(self.losses[name][: n + 1], 1e-16))
        return [a for arts in self.artists.values() for a in arts]

    def show_final(self):
        self.update(self.steps)
        return self.fig

    def animate(self, frames=150, fps=30):
        idx = np.unique(np.linspace(0, self.steps, frames).astype(int))
        return FuncAnimation(self.fig, self.update, frames=idx, interval=1000 / fps, blit=False)


def save_animation(anim, path, fps=30):
    writer = FFMpegWriter(fps=fps) if str(path).endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=80)


def summarize(fn, paths):
    """Print a table of where each optimizer ended up."""
    names = {0: "global min"} | {i: f"local min #{i}" for i in range(1, len(fn.minima))} | {-1: "not converged"}
    print(f"{'optimizer':10s} {'final x':>9s} {'final y':>9s} {'f - f_min':>10s}  result")
    for name, p in paths.items():
        last = p[np.isfinite(p).all(axis=-1)][-1]
        diverged = not np.isfinite(p[-1]).all()
        result = "diverged" if diverged else names[int(fn.which_minimum(last))]
        print(f"{name:10s} {last[0]:9.4f} {last[1]:9.4f} {fn(last) - fn.f_min:10.2e}  {result}")
