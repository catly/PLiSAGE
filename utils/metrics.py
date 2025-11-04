def classification_metrics(y_true, y_pred, threshold=None):
    """
    Computes a dictionary of common binary classification metrics.
    If threshold is None, it finds an optimal threshold based on F1-score.
    """
    from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, confusion_matrix, precision_score, recall_score

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(y_true, y_pred)

    if threshold is None:
        # Find optimal threshold from validation data based on F1 score
        f1_scores = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr) + 1e-8)
        optimal_idx = np.argmax(f1_scores[5:]) + 5 # Start search after first few points
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = threshold

    y_binary_pred = (np.asarray(y_pred) >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_binary_pred)
    cm = confusion_matrix(y_true, y_binary_pred)
    
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    recall = recall_score(y_true, y_binary_pred)
    precision = precision_score(y_true, y_binary_pred)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        'roc_auc': roc_auc,
        'auprc': auprc,
        'f1_score': f1,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold
    }