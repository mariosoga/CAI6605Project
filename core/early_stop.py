class EarlyStopper:
    """Early stopping on a monitored scalar (e.g., val loss or val acc)."""
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = 'min'):
        assert mode in ('min', 'max')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0

    def step(self, current: float) -> bool:
        """Returns True if training should stop."""
        if self.best_value is None:
            self.best_value = current
            return False
        improved = (current < self.best_value - self.min_delta) if self.mode == 'min' else (current > self.best_value + self.min_delta)
        if improved:
            self.best_value = current
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
