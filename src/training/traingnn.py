import torch
import torch.nn.functional as F

from config.settings import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS
)


def train_gnn_model(model, data):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    x_dict = {
        k: v.to(DEVICE)
        for k, v in data.x_dict.items()
    }

    edge_index_dict = {
        k: v.to(DEVICE)
        for k, v in data.edge_index_dict.items()
    }

    y_true = data['order'].y.to(DEVICE)

    mask = data['order'].train_mask.to(DEVICE)

    weights = torch.tensor(
        [1.0, 2.5]
    ).to(DEVICE)

    def train():

        model.train()

        optimizer.zero_grad()

        out_dict = model(
            x_dict,
            edge_index_dict
        )

        loss = F.cross_entropy(
            out_dict['order'][mask],
            y_true[mask]
            # weight=weights
        )

        loss.backward()

        optimizer.step()

        return float(loss)

    print("Iniciando Treinamento GNN...")

    for epoch in range(1, EPOCHS + 1):

        loss = train()

        if epoch % 10 == 0:

            print(
                f'Época: {epoch:03d}, '
                f'Perda: {loss:.4f}'
            )

    return model