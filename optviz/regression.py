"""보고서 3.5: 선형 회귀 y = wx + b 에서 BGD / SGD / mini-batch GD 비교 (개량판)

원본과 달라진 점
- 예측값 배열이 정수형(np.full(..., 0))이라 소수점이 잘리던 버그를 없앴다
- 세 방법은 '배치 크기만 다른 같은 알고리즘'이므로 함수 하나(train)로 합쳤다
  (BGD: batch = n, SGD: batch = 1, mBGD: batch = B)
- 1 epoch = 데이터 전체를 한 번 보는 것으로 통일했다.
  원본은 세 방법 모두 epoch당 12번 업데이트해서, BGD가 전체 데이터를 12번 보는 등 비교 기준이 달랐다
- 정답(최소제곱법 해)을 함께 그려 수렴 여부를 확인할 수 있다
- 파라미터 공간 (w, b) 위의 MSE 등고선에 경로를 그려 4장의 시각화와 연결했다
- seed로 결과를 재현할 수 있고, --report-data로 보고서에 실린 데이터를 그대로 쓸 수 있다
"""
import numpy as np
import matplotlib.pyplot as plt

# 보고서 19쪽 실행 결과로 출력된 데이터
REPORT_X = np.array([2.75398951, 3.54073911, 1.45452369, 2.55413803, 4.46473477, 4.48146544,
                     0.62792655, 1.03621439, 0.25733602, 2.20404922, 0.14938105, 2.28416612])
REPORT_Y = np.array([7.14810379, 11.70683972, 5.44488913, 9.57198714, 13.64423796, 13.23971892,
                     1.53841971, 1.76216586, 1.95437548, 5.71108002, -0.53690336, 6.84684847])

METHOD_COLORS = {"BGD": "red", "SGD": "blue", "mBGD": "green"}


def make_data(n=12, seed=None):
    """y = 3x + 1 + (표준정규분포 오차), x ∈ [0, 5)"""
    rng = np.random.default_rng(seed)
    x = rng.random(n) * 5
    return x, 3 * x + 1 + rng.standard_normal(n)


def mse(w, b, x, y):
    """w, b가 배열이어도 되도록 브로드캐스트한다 (등고선용)"""
    w, b = np.asarray(w)[..., None], np.asarray(b)[..., None]
    return np.mean((w * x + b - y) ** 2, axis=-1)


def gradient(w, b, x, y):
    """L = mean((wx+b - y)^2) 일 때  ∂L/∂w = mean(2(ŷ-y)x),  ∂L/∂b = mean(2(ŷ-y))"""
    r = w * x + b - y
    return np.mean(2 * r * x), np.mean(2 * r)


def train(x, y, batch_size, epochs=100, lr=0.01, w0=0.2, b0=0.8, rng=None):
    """미니 배치 경사 하강법. 업데이트할 때마다 (epoch, w, b)를 기록해서 돌려준다."""
    rng = rng or np.random.default_rng()
    n = len(x)
    w, b = w0, b0
    hist = [(0.0, w, b)]
    for e in range(epochs):
        order = rng.permutation(n)  # 매 epoch마다 데이터를 섞어서 배치로 나눈다
        for s in range(0, n, batch_size):
            batch = order[s:s + batch_size]
            gw, gb = gradient(w, b, x[batch], y[batch])
            w, b = w - lr * gw, b - lr * gb
            hist.append((e + min(s + batch_size, n) / n, w, b))
    return np.array(hist)


def compare(x, y, batch_size=3, epochs=100, lr=0.01, w0=0.2, b0=0.8, seed=None):
    rng = np.random.default_rng(seed)
    runs = {name: train(x, y, bs, epochs, lr, w0, b0, rng)
            for name, bs in [("BGD", len(x)), ("SGD", 1), ("mBGD", batch_size)]}
    w_ols, b_ols = np.polyfit(x, y, 1)  # 최소제곱법으로 구한 정답

    print(f"least squares (exact): w = {w_ols:.4f}, b = {b_ols:.4f}, MSE = {mse(w_ols, b_ols, x, y):.4f}")
    print(f"{'method':6s} {'updates':>8s} {'w':>8s} {'b':>8s} {'MSE':>8s}")
    for name, h in runs.items():
        print(f"{name:6s} {len(h) - 1:8d} {h[-1, 1]:8.4f} {h[-1, 2]:8.4f} {mse(h[-1, 1], h[-1, 2], x, y):8.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_w, ax_b), (ax_loss, ax_path) = axes
    for name, h in runs.items():
        c = METHOD_COLORS[name]
        ax_w.plot(h[:, 0], h[:, 1], color=c, lw=1.2, alpha=0.75, label=name)
        ax_b.plot(h[:, 0], h[:, 2], color=c, lw=1.2, alpha=0.75, label=name)
        ax_loss.plot(h[:, 0], mse(h[:, 1], h[:, 2], x, y) - mse(w_ols, b_ols, x, y) + 1e-12,
                     color=c, lw=1.2, alpha=0.75, label=name)
        ax_path.plot(h[:, 1], h[:, 2], color=c, lw=1, alpha=0.75, marker=".", ms=2, label=name)
    for ax, ref, what in [(ax_w, w_ols, "w"), (ax_b, b_ols, "b")]:
        ax.axhline(ref, color="black", ls="--", lw=1, label=f"least squares {what} = {ref:.3f}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(what)
        ax.set_title(f"{what} during training")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("MSE - MSE(least squares)")
    ax_loss.set_title("loss gap (log scale)")

    # 파라미터 공간 (w, b) 위의 손실 함수 등고선
    allh = np.concatenate(list(runs.values()))
    pad_w, pad_b = np.ptp(allh[:, 1]) * 0.1 + 0.1, np.ptp(allh[:, 2]) * 0.1 + 0.1
    W, B = np.meshgrid(np.linspace(allh[:, 1].min() - pad_w, allh[:, 1].max() + pad_w, 200),
                       np.linspace(allh[:, 2].min() - pad_b, allh[:, 2].max() + pad_b, 200))
    Z = mse(W, B, x, y)
    ax_path.contour(W, B, Z, levels=np.quantile(Z, np.linspace(0, 1, 25) ** 2), colors="grey", linewidths=0.5)
    ax_path.plot(w_ols, b_ols, "*", ms=15, mfc="gold", mec="black", label="least squares")
    ax_path.plot(w0, b0, "s", color="black", label="start")
    ax_path.set_xlabel("w")
    ax_path.set_ylabel("b")
    ax_path.set_title("path in parameter space (contours: MSE)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"BGD vs SGD vs mini-batch GD (batch={batch_size})  —  n={len(x)}, η={lr}, {epochs} epochs")
    fig.tight_layout()
    return fig
