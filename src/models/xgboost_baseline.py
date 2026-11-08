import pandas as pd
from xgboost import XGBClassifier


def make_xgboost_features(df_orders: pd.DataFrame) -> pd.DataFrame:
    dates = df_orders.copy()
    dates["order_month"] = pd.to_datetime(dates["order_date"]).dt.month
    dates["required_month"] = pd.to_datetime(dates["required_date"]).dt.month
    return pd.DataFrame({
        "store_id": dates["store_id"].astype("category"),
        "staff_id": dates["staff_id"].astype("category"),
        "order_month": dates["order_month"].astype("category"),
        "required_month": dates["required_month"].astype("category"),
    })


def train_xgboost_baseline(
    df_orders: pd.DataFrame,
    train_indices,
    validation_indices,
) -> tuple[XGBClassifier, pd.Series, pd.Series]:
    features = make_xgboost_features(df_orders)
    target = df_orders["y"]
    y_train = target.iloc[train_indices]
    y_validation = target.iloc[validation_indices]
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())

    model_kwargs = dict(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42
    )
    if positive:
        model_kwargs["scale_pos_weight"] = negative / positive
    model = XGBClassifier(**model_kwargs, enable_categorical=True)
    model.fit(features.iloc[train_indices], y_train)
    return (
        model,
        y_validation,
        model.predict_proba(features.iloc[validation_indices])[:, 1],
    )