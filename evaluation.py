# Core libraries
import torch

# Data manipulation and visualization

import numpy as np
import matplotlib.pyplot as plt


# Machine learning utilities
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Progress tracking and utilities
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Get Confusion Matrix for each model
for name, model in list(transfer_models.items())[:2]:
    print(name)
    all_preds = []
    all_labels = []

    # Disable gradient tracking during inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) # Get predicted class
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names= CLASSES)
    print(cr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES) # class_names is a list of your class labels
    disp.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()