import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_seasonality(df_orders):

    df_orders['order_date'] = pd.to_datetime(
        df_orders['order_date']
    )

    sazonal = df_orders.groupby(
        df_orders['order_date'].dt.to_period('M')
    )['y'].mean()

    plt.figure(figsize=(10, 4))

    sazonal.plot(
        kind='line',
        marker='o',
        color='teal'
    )

    plt.title('Taxa de Atrasos por Mês')

    plt.show()


def plot_imbalance(df_orders):

    plt.figure(figsize=(6, 4))

    sns.countplot(x=df_orders['y'])

    plt.title(
        'Distribuição: '
        'No Prazo (0) vs Atrasado (1)'
    )

    plt.show()