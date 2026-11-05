# Optimizer Visualize

Visualizes how gradient-based optimizers (GD, Momentum, NAG, Adagrad, RMSProp, Adam, Nadam) move across a 2D loss surface.
Based on a 2024 high-school math research report, “The Review of Optimizer in Deep Learning”.

**Interactive version (no install):** https://camo1875.github.io/optviz/

![Optimizer paths on the report's loss function](results/landscape_report.gif)

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Plots

Each command opens a matplotlib window. Add `--save file.png` (or `.gif` / `.mp4` for animations) to write a file instead.

```bash
python -m optviz landscape                 # optimizer paths: 3D surface, top view, loss gap
python -m optviz landscape --animate       # same, animated
python -m optviz regression --report-data  # BGD vs SGD vs mini-batch GD on linear regression
python -m optviz beta1                     # Adam with different β1
python -m optviz basin                     # where each optimizer ends up, for every start point
python -m optviz hparam                    # where Adam ends up, for every (learning rate, β1)
python -m optviz all                       # save all of the above to results/
```

Common options for `landscape`:

| option | meaning |
|---|---|
| `-f {report,bowl,saddle,rosenbrock}` | loss function (default `report`) |
| `-o GD,Momentum,Adam` | optimizers to compare |
| `--start X Y` | start point |
| `--lr 0.01` | learning rate |
| `--beta1 0.99` | Adam/Nadam β1 |
| `--steps 1000` | number of updates |

Run `python -m optviz <command> -h` for every option.

The code as printed in the report is in `original/` and runs on its own:

```bash
python original/ch3_gradient_descent_compare.py
python original/ch4_optimizer_visualize.py
```
