import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from config.settings import DEVICE


def evaluate_gnn(model, data):

    model.eval()

    with torch.no_grad():

        x_dict_gpu = {
            k: v.to(DEVICE)
            for k, v in data.x_dict.items()
        }

        edge_index_dict_gpu = {
            k: v.to(DEVICE)
            for k, v in data.edge_index_dict.items()
        }

        mask = data['order'].test_mask.to(DEVICE)

        target = data['order'].y.to(DEVICE)

        out_dict = model(
            x_dict_gpu,
            edge_index_dict_gpu
        )

        logits = out_dict['order'][mask]

        preds = logits.argmax(dim=-1).cpu().numpy()

        y_true = target[mask].cpu().numpy()

    return y_true, preds


def print_gnn_metrics(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)

    print("\n--- Resultados Finais GNN ---")

    print(f"Acurácia Geral: {acc:.2%}")

    print("\nRelatório de Classificação:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=['No Prazo', 'Atrasado'],
            zero_division=0
        )
    )


def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=['No Prazo', 'Atrasado'],
        yticklabels=['No Prazo', 'Atrasado']
    )

    plt.xlabel('Predição')
    plt.ylabel('Realidade')

    plt.title(
        'Matriz de Confusão - '
        'Predição de Atrasos'
    )

    plt.show()