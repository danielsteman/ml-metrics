import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

y_true = np.array([[1, 0, 1],[0, 1, 0]])
y_pred = np.array([[1, 0, 0],[0, 1, 1]])

matrix = multilabel_confusion_matrix(y_true, y_pred)