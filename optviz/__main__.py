"""사용법:  python -m optviz <명령> [옵션]      (명령별 옵션은 python -m optviz <명령> -h)

  landscape   손실 함수 위에서 옵티마이저 경로 비교 (보고서 4.6)   --animate 로 애니메이션
  regression  선형 회귀에서 BGD / SGD / mBGD 비교 (보고서 3.5)
  beta1       Adam의 β1에 따른 경로 비교 (보고서 37쪽)
  basin       모든 시작점에 대해 각 옵티마이저가 어디로 가는지 지도
  hparam      학습률 × β1 조합별로 Adam이 어디로 가는지 지도
  all         위 그림들을 모두 results/ 에 저장
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt

from . import experiments, regression
from . import optimizers as O
from .functions import FUNCTIONS
from .visualize import Scene, save_animation, summarize


def finish(fig, save):
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"saved: {save}")
        plt.close(fig)
    else:
        plt.show()


def cmd_landscape(a):
    fn = FUNCTIONS[a.function]
    start = a.start or fn.start
    steps = a.steps or fn.steps
    paths = {}
    for name in a.optimizers.split(","):
        kw = {"beta1": a.beta1} if a.beta1 is not None and name in ("Adam", "Nadam") else {}
        paths[name] = O.run(O.make(name, fn, lr=a.lr, **kw), fn, start, steps)
    summarize(fn, paths)
    scene = Scene(fn, paths)
    if not a.animate:
        return finish(scene.show_final(), a.save)
    anim = scene.animate(frames=a.frames)
    if a.save:
        save_animation(anim, a.save)
        print(f"saved: {a.save}")
    else:
        plt.show()


def cmd_regression(a):
    if a.report_data:
        x, y = regression.REPORT_X, regression.REPORT_Y
    else:
        x, y = regression.make_data(a.n, a.seed)
    finish(regression.compare(x, y, a.batch_size, a.epochs, a.lr, seed=a.seed), a.save)


def cmd_beta1(a):
    fn = FUNCTIONS[a.function]
    finish(experiments.beta1_paths(fn, a.betas, a.lr, a.steps, a.start), a.save)


def cmd_basin(a):
    fn = FUNCTIONS[a.function]
    finish(experiments.basin_map(fn, a.optimizers.split(","), a.res, a.steps), a.save)


def cmd_hparam(a):
    fn = FUNCTIONS[a.function]
    finish(experiments.hparam_map(fn, a.optimizer, steps=a.steps, start=a.start), a.save)


def cmd_all(a):
    out = a.out
    run = lambda argv: main(argv)  # noqa: E731
    run(["regression", "--report-data", "--seed", "0", "--save", f"{out}/regression_report_data.png"])
    run(["regression", "--seed", "0", "--n", "100", "--batch-size", "10", "--save", f"{out}/regression_n100.png"])
    for key in FUNCTIONS:
        run(["landscape", "-f", key, "--save", f"{out}/landscape_{key}.png"])
    run(["landscape", "-f", "report", "--beta1", "0.99", "--save", f"{out}/landscape_report_adam_beta0.99.png"])
    run(["landscape", "-f", "report", "--animate", "--save", f"{out}/landscape_report.gif"])
    run(["beta1", "--save", f"{out}/beta1_report.png"])
    run(["basin", "--save", f"{out}/basin_report.png"])
    run(["hparam", "--save", f"{out}/hparam_adam_report.png"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m optviz", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, steps=True):
        sp.add_argument("-f", "--function", choices=list(FUNCTIONS), default="report", help="손실 함수 (기본: report)")
        sp.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="시작점 (기본: 함수별 기본값)")
        if steps:
            sp.add_argument("--steps", type=int, help="반복 횟수 (기본: 함수별 기본값)")
        sp.add_argument("--save", help="그림 저장 경로 (.png / 애니메이션은 .gif, .mp4). 없으면 창으로 띄움")

    s = sub.add_parser("landscape", help="옵티마이저 경로 비교")
    common(s)
    s.add_argument("-o", "--optimizers", default=",".join(O.REPORT_OPTIMIZERS),
                   help=f"쉼표로 구분 (가능: {','.join(O.OPTIMIZERS)})")
    s.add_argument("--lr", type=float, help="모든 옵티마이저의 학습률 (기본: 함수별 기본값)")
    s.add_argument("--beta1", type=float, help="Adam/Nadam의 β1 (기본 0.9)")
    s.add_argument("--animate", action="store_true", help="애니메이션으로 보기/저장")
    s.add_argument("--frames", type=int, default=150, help="애니메이션 프레임 수")
    s.set_defaults(func=cmd_landscape)

    s = sub.add_parser("regression", help="선형 회귀에서 BGD/SGD/mBGD 비교")
    s.add_argument("--report-data", action="store_true", help="보고서에 실린 12개 데이터 사용")
    s.add_argument("--n", type=int, default=12, help="데이터 개수")
    s.add_argument("--batch-size", type=int, default=3)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--lr", type=float, default=0.01)
    s.add_argument("--seed", type=int, help="난수 시드 (재현용)")
    s.add_argument("--save")
    s.set_defaults(func=cmd_regression)

    s = sub.add_parser("beta1", help="Adam β1별 경로")
    common(s)
    s.add_argument("--betas", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    s.add_argument("--lr", type=float)
    s.set_defaults(func=cmd_beta1)

    s = sub.add_parser("basin", help="시작점별 도착 지도")
    s.add_argument("-f", "--function", choices=list(FUNCTIONS), default="report")
    s.add_argument("-o", "--optimizers", default=",".join(O.REPORT_OPTIMIZERS))
    s.add_argument("--res", type=int, default=121, help="격자 해상도")
    s.add_argument("--steps", type=int)
    s.add_argument("--save")
    s.set_defaults(func=cmd_basin)

    s = sub.add_parser("hparam", help="학습률 × β1 지도")
    common(s)
    s.add_argument("--optimizer", default="Adam", choices=["Adam", "Nadam"])
    s.set_defaults(func=cmd_hparam)

    s = sub.add_parser("all", help="모든 그림을 저장")
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_all)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
