"""Run one of the Lightning training scripts unchanged and print plain progress lines, as a training loop would:

  [train] lewm_antl | epoch 3/10 | step 1200/3000 (40%) | loss 0.0123 | 2.6 it/s | epoch left 0:11:32 | training left 1:32:10
  [epoch] lewm_antl | epoch 3/10 done in 0:19:20 | loss 0.0119 | val_loss 0.0131 | training left 1:13:05

A line every PROGRESS_EVERY seconds (default 30) and one at the end of every epoch, so `tail -f` on the log (or
scripts/hpc/progress.sh) shows where the training is and how long it still needs.  The script itself is not modified: every
lightning Trainer it creates gets one extra callback.  Scripts that train in two phases (GCIVL, GCIQL, HILP: value, then policy)
report each phase on its own, numbered.

usage: python train_progress.py <label> <training script.py> [its Hydra overrides ...]
"""
import os
import runpy
import sys
import time

import lightning.pytorch as pl


def hms(s):
    s = int(max(s, 0)); return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'


def _num(x):
    try:
        return float(x.detach().float().mean()) if hasattr(x, 'detach') else float(x)
    except Exception:  # noqa: BLE001
        return None


class ProgressLog(pl.Callback):
    def __init__(self, label, phase):
        self.label = label + (f' phase {phase}' if phase > 1 else ''); self.every = float(os.environ.get('PROGRESS_EVERY', 30))
        self.t0 = self.te = self.last = time.time(); self.done = 0; self.loss = None

    def _total(self, trainer):
        n = trainer.num_training_batches
        return int(n) if n not in (None, float('inf')) else None

    def _left(self, trainer):
        n, rate = self._total(trainer), self.done / max(time.time() - self.t0, 1e-9)
        if not n or not trainer.max_epochs or trainer.max_epochs < 0 or rate <= 0:
            return None, rate
        return (n * trainer.max_epochs - self.done) / rate, rate

    def on_train_start(self, trainer, module):
        self.t0 = self.te = self.last = time.time(); self.done = 0
        n = self._total(trainer)
        print(f'[train] {self.label} | start: {trainer.max_epochs} epochs x {n if n else "?"} steps', flush=True)

    def on_train_epoch_start(self, trainer, module):
        self.te = time.time()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.done += 1
        v = _num(outputs.get('loss')) if isinstance(outputs, dict) else _num(outputs) if outputs is not None else None
        if v is None:   # manual optimisation: take a logged loss
            v = next((_num(x) for k, x in trainer.callback_metrics.items() if 'loss' in k.lower()), None)
        self.loss = v if v is not None else self.loss
        if time.time() - self.last < self.every:
            return
        self.last = time.time(); n = self._total(trainer); left, rate = self._left(trainer)
        step = f'step {batch_idx + 1}/{n} ({100 * (batch_idx + 1) / n:.0f}%)' if n else f'step {batch_idx + 1}'
        tail = f' | epoch left {hms((n - batch_idx - 1) / rate)} | training left {hms(left)}' if left is not None else ''
        print(f'[train] {self.label} | epoch {trainer.current_epoch + 1}/{trainer.max_epochs} | {step} | '
              f'loss {self.loss if self.loss is None else format(self.loss, ".4g")} | {rate:.2f} it/s{tail}', flush=True)

    def on_train_epoch_end(self, trainer, module):
        m = {k: _num(x) for k, x in trainer.callback_metrics.items() if 'loss' in k.lower()}
        main = {k: x for k, x in m.items() if k.lower().split('/')[-1] in ('loss', 'val_loss', 'train_loss')} or dict(sorted(m.items())[:3])
        ms = ' | '.join(f'{k} {x:.4g}' for k, x in sorted(main.items()) if x is not None)
        left, _ = self._left(trainer)
        print(f'[epoch] {self.label} | epoch {trainer.current_epoch + 1}/{trainer.max_epochs} done in {hms(time.time() - self.te)}'
              f'{" | " + ms if ms else ""}{" | training left " + hms(left) if left is not None else ""}', flush=True)

    def on_train_end(self, trainer, module):
        print(f'[train] {self.label} | finished in {hms(time.time() - self.t0)}', flush=True)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    label, script = sys.argv[1], os.path.abspath(sys.argv[2])
    count = [0]; orig = pl.Trainer.__init__

    def init(self, *a, **k):
        count[0] += 1
        k['callbacks'] = list(k.get('callbacks') or []) + [ProgressLog(label, count[0])]
        orig(self, *a, **k)

    pl.Trainer.__init__ = init
    sys.argv = [script] + sys.argv[3:]
    sys.path.insert(0, os.path.dirname(script))   # as `python script.py` would: imports next to the script, Hydra's ./config
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
