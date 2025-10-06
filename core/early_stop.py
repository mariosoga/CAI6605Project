class EarlyStopper:
    """
    Early stopping on a monitored scalar (e.g., val loss or val acc).

    Backward-compatible with your original:
      - __init__(patience=3, min_delta=0.0, mode='min')
      - step(current) -> bool   # True => stop

    Extras (optional):
      - rel_delta: treat min_delta as a *fraction* of |best| (e.g., 0.01 = 1%)
      - cooldown: after an improvement, wait N steps before counting "bad" epochs
      - initial_best: seed the best value explicitly (useful when resuming)
      - best_epoch: track which epoch achieved best_value
      - state_dict()/load_state_dict() for checkpointing the stopper itself
      - reset() to start over
    """
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = 'min',           # 'min' for loss, 'max' for accuracy
        rel_delta: bool = False,     # if True, threshold = |best| * min_delta
        cooldown: int = 0,           # epochs to ignore "bad" count after an improvement
        initial_best: float | None = None
    ):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.rel_delta = bool(rel_delta)
        self.cooldown = int(cooldown)

        self.best_value: float | None = initial_best
        self.best_epoch: int | None = None

        self._bad_count = 0
        self._cooldown_left = 0
        self._stopped = False
        self._seen_steps = 0

    def _threshold(self, best: float) -> float:
        if not self.rel_delta:
            return self.min_delta
        # relative threshold, scaled by |best| (handles sign)
        return abs(best) * self.min_delta

    def _is_improved(self, current: float) -> bool:
        if self.best_value is None:
            return True
        th = self._threshold(self.best_value)
        if self.mode == 'min':
            return current < (self.best_value - th)
        else:  # 'max'
            return current > (self.best_value + th)

    def step(self, current: float, epoch: int | None = None) -> bool:
        """
        Update with the current metric value. Returns True if training should stop.
        """
        self._seen_steps += 1

        if self._is_improved(current):
            self.best_value = float(current)
            self.best_epoch = int(epoch) if epoch is not None else self._seen_steps
            self._bad_count = 0
            self._cooldown_left = self.cooldown
            self._stopped = False
            return False

        # Not improved
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            # during cooldown we don't increment bad_count
            return False

        self._bad_count += 1
        if self._bad_count >= self.patience:
            self._stopped = True
        return self._stopped

    @property
    def stopped(self) -> bool:
        return self._stopped

    def reset(self):
        self.best_value = None
        self.best_epoch = None
        self._bad_count = 0
        self._cooldown_left = 0
        self._stopped = False
        self._seen_steps = 0

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "rel_delta": self.rel_delta,
            "cooldown": self.cooldown,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_count": self._bad_count,
            "cooldown_left": self._cooldown_left,
            "stopped": self._stopped,
            "seen_steps": self._seen_steps,
        }

    def load_state_dict(self, state: dict):
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.mode = state.get("mode", self.mode)
        self.rel_delta = bool(state.get("rel_delta", self.rel_delta))
        self.cooldown = int(state.get("cooldown", self.cooldown))
        self.best_value = state.get("best_value", self.best_value)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        self._bad_count = int(state.get("bad_count", self._bad_count))
        self._cooldown_left = int(state.get("cooldown_left", self._cooldown_left))
        self._stopped = bool(state.get("stopped", self._stopped))
        self._seen_steps = int(state.get("seen_steps", self._seen_steps))

    def __repr__(self) -> str:
        return (f"EarlyStopper(mode={self.mode!r}, patience={self.patience}, "
                f"min_delta={self.min_delta}, rel_delta={self.rel_delta}, cooldown={self.cooldown}, "
                f"best_value={self.best_value}, best_epoch={self.best_epoch}, "
                f"bad_count={self._bad_count}, stopped={self._stopped})")

