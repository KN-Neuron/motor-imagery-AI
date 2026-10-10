"""Does each model learn something that transfers to validation? One line per (run, pipeline/experiment).

    python scripts/inspect_history.py results/final

Columns (averaged over fits = folds x seeds): best_ep = median best epoch (n0 = fits with best epoch 0),
va_ep0 / va_best / va_max / va_last = validation accuracy at epoch 0, at the selected epoch, the maximum
over epochs and at the last epoch; tr_last = training accuracy at the last epoch.
va_max close to va_ep0 means validation accuracy never improved (no transferable learning).
Writes <root>/history_summary.csv.
"""
import sys
from pathlib import Path

import pandas as pd


def summarize(h: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for g, d in h.groupby(key, sort=False):
        fits = []
        for _, f in d.groupby(["fold", "seed"]):
            f = f.sort_values("epoch")
            be = int(f.best_epoch.iloc[0])
            fits.append(dict(best=be, ep0=f.val_acc.iloc[0], at_best=f.val_acc[f.epoch == be].iloc[0],
                             vmax=f.val_acc.max(), last=f.val_acc.iloc[-1], tr=f.train_acc.iloc[-1]))
        F = pd.DataFrame(fits)
        rows.append(dict(group=g, fits=len(F), best_ep=F.best.median(), n0=int((F.best == 0).sum()),
                         va_ep0=100 * F.ep0.mean(), va_best=100 * F.at_best.mean(), va_max=100 * F.vmax.mean(),
                         va_last=100 * F["last"].mean(), tr_last=100 * F.tr.mean()))
    return pd.DataFrame(rows)


def main(root):
    root = Path(root)
    out = []
    for f in sorted(root.glob("*/history.csv")):
        h = pd.read_csv(f)
        key = "experiment" if "experiment" in h else "pipeline"
        out.append(summarize(h, key).assign(run=f.parent.name))
    if not out:
        print(f"no history.csv under {root}")
        return
    df = pd.concat(out)[["run", "group", "fits", "best_ep", "n0", "va_ep0", "va_best", "va_max", "va_last", "tr_last"]]
    df.to_csv(root / "history_summary.csv", index=False)
    print(f"{'run':<28} {'pipeline / experiment':<44} {'fits':>4} {'best_ep':>7} {'n0':>3} "
          f"{'va_ep0':>6} {'va_best':>7} {'va_max':>6} {'va_last':>7} {'tr_last':>7}")
    for r in df.itertuples():
        print(f"{r.run:<28} {r.group:<44} {r.fits:>4} {r.best_ep:>7.0f} {r.n0:>3} {r.va_ep0:>6.1f} "
              f"{r.va_best:>7.1f} {r.va_max:>6.1f} {r.va_last:>7.1f} {r.tr_last:>7.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/final")
