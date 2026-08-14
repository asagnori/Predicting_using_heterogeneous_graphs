import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_schema(
    data,
    output_path="schema_grafo.png"
):
    try:

        G = nx.DiGraph()

        # =====================================================
        # Adiciona relações
        # =====================================================

        for src, rel, dst in data.edge_types:
            G.add_edge(
                src,
                dst,
                label=rel
            )

        # =====================================================
        # Layout
        # =====================================================

        pos = nx.spring_layout(
            G,
            seed=42
        )
        plt.figure(figsize=(10, 7))

        # =====================================================
        # Nós e arestas
        # =====================================================

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3500,
            font_size=10,
            arrows=True
        )

        # =====================================================
        # Labels das arestas
        # =====================================================
        edge_labels = nx.get_edge_attributes(
            G,
            "label"
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels
        )

        # =====================================================
        # Título
        # =====================================================

        plt.title(
            "Schema do Grafo Heterogêneo"
        )
        plt.savefig(
            output_path,
            bbox_inches="tight"
        )
        plt.show()
        print(
            f"Schema salvo em: {output_path}"
        )

    except Exception as e:
        print(
            f"Erro ao gerar visualização: {e}"
        )
        print(
            "Instale: pip install networkx matplotlib"
        )