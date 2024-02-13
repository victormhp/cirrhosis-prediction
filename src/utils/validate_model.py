def validate_model(pipeline, X, y, n_splits=5, n_repeats=1):
    import numpy as np
    from sklearn.metrics import log_loss
    from sklearn.model_selection import RepeatedStratifiedKFold

    train_scores, val_scores = [], []
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    for (train_idx, val_idx) in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_preds = pipeline.predict_proba(X_train)
        valid_preds = pipeline.predict_proba(X_val)

        train_score = log_loss(y_train, train_preds)
        val_score = log_loss(y_val, valid_preds)

        train_scores.append(train_score)
        val_scores.append(val_score)

    avg_train_score = np.mean(train_scores)
    avg_val_score = np.mean(val_scores)

    return avg_train_score, avg_val_score
