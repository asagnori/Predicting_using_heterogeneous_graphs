import pandas as pd
from xgboost import XGBClassifier

def train_xgboost_baseline(df_orders):

    df_xgb = df_orders.copy()

    # =====================================================
    # Features
    # =====================================================

    df_xgb['order_month'] = pd.to_datetime(
        df_xgb['order_date']
    ).dt.month

    df_xgb['required_month'] = pd.to_datetime(
        df_xgb['required_date']
    ).dt.month

    X = pd.DataFrame({

        'customer_id':
            df_xgb['customer_id'],

        'store_id':
            df_xgb['store_id'],

        'staff_id':
            df_xgb['staff_id'],

        'order_month':
            df_xgb['order_month'],

        'required_month':
            df_xgb['required_month']

    })

    y = df_xgb['y']

    # =====================================================
    # Split Temporal
    # =====================================================

    train_mask_xgb = (
        pd.to_datetime(df_xgb['order_date'])
        < '2018-01-01'
    )

    test_mask_xgb = (
        pd.to_datetime(df_xgb['order_date'])
        >= '2018-01-01'
    )

    X_train = X[train_mask_xgb]
    X_test = X[test_mask_xgb]

    y_train = y[train_mask_xgb]
    y_test = y[test_mask_xgb]

    # =====================================================
    # Modelo
    # =====================================================

    model = XGBClassifier(

        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42

    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nBaseline XGBoost treinado.")

    return y_test, y_pred