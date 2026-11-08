import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


def print_comparative_table(

    y_test_xgb,
    y_pred_xgb,
    y_true_gnn,
    y_pred_gnn,
    xgb_probabilities=None,
    gnn_probabilities=None,
    y_pred_dummy=None,
    dummy_probabilities=None,

):

    y_true_gnn = np.array(y_true_gnn)
    y_pred_gnn = np.array(y_pred_gnn)

    # =====================================================
    # Métricas XGBoost
    # =====================================================

    acc_xgb = accuracy_score(
        y_test_xgb,
        y_pred_xgb
    )

    prc_xgb = precision_score(
        y_test_xgb,
        y_pred_xgb,
        zero_division=0
    )

    rec_xgb = recall_score(
        y_test_xgb,
        y_pred_xgb,
        zero_division=0
    )

    f1_xgb = f1_score(
        y_test_xgb,
        y_pred_xgb,
        zero_division=0
    )

    # =====================================================
    # Métricas GNN
    # =====================================================

    acc_gnn = accuracy_score(
        y_true_gnn,
        y_pred_gnn
    )

    prc_gnn = precision_score(
        y_true_gnn,
        y_pred_gnn,
        zero_division=0
    )

    rec_gnn = recall_score(
        y_true_gnn,
        y_pred_gnn,
        zero_division=0
    )

    f1_gnn = f1_score(
        y_true_gnn,
        y_pred_gnn,
        zero_division=0
    )
    xgb_scores = (
        y_pred_xgb if xgb_probabilities is None else xgb_probabilities
    )
    gnn_scores = (
        y_pred_gnn if gnn_probabilities is None else gnn_probabilities
    )
    pr_auc_xgb = average_precision_score(y_test_xgb, xgb_scores)
    pr_auc_gnn = average_precision_score(y_true_gnn, gnn_scores)

    # =====================================================
    # Impressão
    # =====================================================

    print("\n" + "="*70)

    print(
        f"{'METRIC (CLASS: DELAYED)':<30} | "
        f"{'XGBOOST BASELINE':<16} | "
        f"{'GRAPH-SAGE GNN':<16}"
    )

    print("-"*70)

    print(
        f"{'Global Accuracy (Acurácia)':<30} | "
        f"{acc_xgb:<16.2%} | "
        f"{acc_gnn:<16.2%}"
    )

    print(
        f"{'Class Precision (Precisão)':<30} | "
        f"{prc_xgb:<16.4f} | "
        f"{prc_gnn:<16.4f}"
    )

    print(
        f"{'Class Recall (Revocação)':<30} | "
        f"{rec_xgb:<16.4f} | "
        f"{rec_gnn:<16.4f}"
    )

    print(
        f"{'F1-Score (Medida F)':<30} | "
        f"{f1_xgb:<16.4f} | "
        f"{f1_gnn:<16.4f}"
    )
    print(
        f"{'PR-AUC':<30} | "
        f"{pr_auc_xgb:<16.4f} | "
        f"{pr_auc_gnn:<16.4f}"
    )
    if y_pred_dummy is not None:
        dummy_scores = (
            y_pred_dummy
            if dummy_probabilities is None
            else dummy_probabilities
        )
        print(
            f"{'Dummy PR-AUC':<30} | "
            f"{average_precision_score(y_test_xgb, dummy_scores):<16.4f}"
        )

    print("-"*70)

    base_prev_xgb = np.mean(y_test_xgb)
    base_prev_gnn = np.mean(y_true_gnn)

    print(
        f"{'Target Base Prevalence':<30} | "
        f"{base_prev_xgb:<16.2%} | "
        f"{base_prev_gnn:<16.2%}"
    )

    print(
        f"{'Total Test Instances (2018)':<30} | "
        f"{len(y_test_xgb):<16} | "
        f"{len(y_true_gnn):<16}"
    )

    print("="*70)