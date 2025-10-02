import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
