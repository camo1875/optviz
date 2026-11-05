"""Usage:  python -m optviz <command> [options]      (options per command: python -m optviz <command> -h)

  landscape   compare optimizer paths on a loss surface (report 4.6); --animate for an animation
  regression  BGD / SGD / mini-batch GD on linear regression (report 3.5)
  beta1       Adam paths for different β1 (report p.37)
  basin       map of where each optimizer ends up, for every start point
  hparam      map of where Adam ends up, for every (learning rate, β1) pair
  all         save all of the figures above to results/
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
        sp.add_argument("-f", "--function", choices=list(FUNCTIONS), default="report", help="loss function (default: report)")
        sp.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="start point (default: per function)")
        if steps:
            sp.add_argument("--steps", type=int, help="number of updates (default: per function)")
        sp.add_argument("--save", help="save to this path (.png; animations .gif or .mp4) instead of opening a window")

    s = sub.add_parser("landscape", help="compare optimizer paths")
    common(s)
    s.add_argument("-o", "--optimizers", default=",".join(O.REPORT_OPTIMIZERS),
                   help=f"comma-separated (available: {','.join(O.OPTIMIZERS)})")
    s.add_argument("--lr", type=float, help="learning rate for every optimizer (default: per function)")
    s.add_argument("--beta1", type=float, help="β1 for Adam/Nadam (default 0.9)")
    s.add_argument("--animate", action="store_true", help="show or save an animation")
    s.add_argument("--frames", type=int, default=150, help="number of animation frames")
    s.set_defaults(func=cmd_landscape)

    s = sub.add_parser("regression", help="BGD/SGD/mBGD on linear regression")
    s.add_argument("--report-data", action="store_true", help="use the 12 data points printed in the report")
    s.add_argument("--n", type=int, default=12, help="number of data points")
    s.add_argument("--batch-size", type=int, default=3)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--lr", type=float, default=0.01)
    s.add_argument("--seed", type=int, help="random seed (for reproducibility)")
    s.add_argument("--save")
    s.set_defaults(func=cmd_regression)

    s = sub.add_parser("beta1", help="Adam paths for different β1")
    common(s)
    s.add_argument("--betas", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    s.add_argument("--lr", type=float)
    s.set_defaults(func=cmd_beta1)

    s = sub.add_parser("basin", help="where each optimizer ends up, per start point")
    s.add_argument("-f", "--function", choices=list(FUNCTIONS), default="report")
    s.add_argument("-o", "--optimizers", default=",".join(O.REPORT_OPTIMIZERS))
    s.add_argument("--res", type=int, default=121, help="grid resolution")
    s.add_argument("--steps", type=int)
    s.add_argument("--save")
    s.set_defaults(func=cmd_basin)

    s = sub.add_parser("hparam", help="learning rate × β1 map")
    common(s)
    s.add_argument("--optimizer", default="Adam", choices=["Adam", "Nadam"])
    s.set_defaults(func=cmd_hparam)

    s = sub.add_parser("all", help="save every figure")
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_all)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
