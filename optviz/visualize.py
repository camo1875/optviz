"""옵티마이저 경로 시각화: 3D 곡면 + 2D 등고선 + 손실 그래프를 한 화면에.

원본(4.6)과 달라진 점
- 경로가 실제로 움직이는 영역만 그려서 경로가 잘 보인다 (원본은 [-5,5]라 z가 600까지 커졌다)
- 위에서 내려다본 등고선과 최솟값/극솟값/안장점 표시를 추가했다
- 스텝별 f(Θ_n) - f_min 을 로그 스케일로 그려서 수렴 속도를 비교할 수 있다
- plt.ion + time.sleep 대신 FuncAnimation을 써서 GIF/MP4로 저장할 수 있다
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
    """등고선과 임계점을 그린다. 다른 그림(experiments.py)에서도 쓴다."""
    gx, gy, z = grid(fn, res)
    # 낮은 곳의 등고선이 촘촘하도록 분위수로 높이를 고른다
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
    """경로들을 n번째 스텝까지 그리는 그림. update(n)을 바꿔가며 부르면 애니메이션이 된다."""

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
    """각 옵티마이저가 어디에 도착했는지 표로 출력한다."""
    names = {0: "global min"} | {i: f"local min #{i}" for i in range(1, len(fn.minima))} | {-1: "not converged"}
    print(f"{'optimizer':10s} {'final x':>9s} {'final y':>9s} {'f - f_min':>10s}  result")
    for name, p in paths.items():
        last = p[np.isfinite(p).all(axis=-1)][-1]
        diverged = not np.isfinite(p[-1]).all()
        result = "diverged" if diverged else names[int(fn.which_minimum(last))]
        print(f"{name:10s} {last[0]:9.4f} {last[1]:9.4f} {fn(last) - fn.f_min:10.2e}  {result}")
