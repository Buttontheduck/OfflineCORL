import wandb
from collections import defaultdict
import numpy as np

class MetricsLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsLogger, cls).__new__(cls)
            # Use lists to store values in case a metric is calculated 
            # multiple times (e.g., inside a mini-batch loop) before the step ends
            cls._instance.buffer = defaultdict(list)
        return cls._instance

    def log(self, key: str, value: float):
        """Add a scalar to the buffer."""
        self.buffer[key].append(value)

    def dump(self, step: int):
        """Average the buffered metrics, push to wandb, and clear the buffer."""
        if not self.buffer:
            return

        # Average the values for any metric logged multiple times in one step
        aggregated_metrics = {
            key: np.mean(values) for key, values in self.buffer.items()
        }

        # Single wandb call ensures perfectly synchronized steps
        wandb.log(aggregated_metrics, step=step)

        # Wipe the buffer clean for the next environment step
        self.buffer.clear()

# Instantiate it here so other files can import this exact instance
global_logger = MetricsLogger()