import os

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

from src.config.settings import DATABASE_CONFIG
from src.data.loader import load_data
from src.explainability.gnn import create_order_explainer, explain_orders
from src.explainability.shap import explain_xgboost
from src.graph.graph_builder import build_graph
from src.training.cross_validation import (
    run_temporal_experiment,
    temporal_results_table,
)


def create_database_engine():
    url = URL.create(
        "mysql+pymysql",
        username=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
        host=DATABASE_CONFIG["host"].split(":")[0],
        port=int(DATABASE_CONFIG["port"])
        if DATABASE_CONFIG["port"]
        else (
            int(DATABASE_CONFIG["host"].split(":")[1])
            if ":" in DATABASE_CONFIG["host"]
            else None
        ),
        database=DATABASE_CONFIG["database"],
    )
    return create_engine(url)


def main() -> None:
    try:
        engine = create_database_engine()
        df_dict = load_data(engine)
    except OperationalError as error:
        if "1045" in str(error):
            raise RuntimeError(
                "MySQL recusou as credenciais. Verifique DB_USER, "
                "DB_PASSWORD, DB_HOST e DB_PORT no arquivo .env e confirme "
                "que o usuário possui acesso ao banco bike_store."
            ) from error
        raise
    data, df_orders, maps = build_graph(df_dict)
    print(data)
    print(f"Mapeamentos criados: {', '.join(maps)}")
    print(f"Pedidos preparados: {len(df_orders)}")

    folds = int(os.getenv("CV_FOLDS", "4"))
    epochs = int(os.getenv("GNN_EPOCHS", "50"))
    print(
        f"\nIniciando experimento temporal: {folds} folds, "
        f"{epochs} épocas GNN por fold..."
    )
    results = run_temporal_experiment(
        data,
        df_orders,
        n_splits=folds,
        gnn_epochs=epochs,
    )
    results_table = temporal_results_table(results)
    print("\nResultados por fold:")
    print(results_table.to_string(index=False))
    print("\nMédia das métricas:")
    print(
        results_table.groupby("model", sort=True)[
            ["accuracy", "precision", "recall", "f1", "pr_auc"]
        ].mean().to_string()
    )

    last_fold = results[-1]
    _, shap_features = explain_xgboost(
        last_fold["xgb_model"],
        df_orders.iloc[last_fold["train_indices"]],
    )
    print(
        "\nSHAP XGBoost calculado com as features: "
        + ", ".join(shap_features.columns)
    )

    if os.getenv("RUN_EXPLAINER", "1").lower() not in {"0", "false", "no"}:
        explainer_epochs = int(os.getenv("EXPLAINER_EPOCHS", "50"))
        order_indices = last_fold["validation_indices"][:1]
        print(
            f"Iniciando GNNExplainer para {len(order_indices)} pedido(s), "
            f"{explainer_epochs} épocas..."
        )
        explainer = create_order_explainer(
            last_fold["gnn_model"],
            epochs=explainer_epochs,
        )
        explanations = explain_orders(
            explainer,
            last_fold["x_dict"],
            last_fold["edge_index"],
            order_indices,
        )
        print(f"Relações explicadas pelo GNNExplainer: {len(explanations)}")


if __name__ == "__main__":
    main()