"""Display functions for the random walk ranking algorithm."""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def display_results(results):
    """Displays the random walk algorithm results in a dataframe."""
    rows = []
    for graph_name, graph_results in results.items():
        for d, d_results in graph_results.items():
            top5 = [int(node) for node in d_results["top5"]]
            top5_vals = [f"{float(v):.4f}" for v in d_results["pr"][d_results["top5"]]]

            row = {
                "Graph": graph_name,
                "Vertices": d_results["vertices"],
                "Edges": d_results["edges"],
                "Damping factor": d,
                "Iterations": d_results["iterations"],
                "Time": f"{d_results['time']:.6f}s",
                "Top 5 nodes": str(top5),
                "Ranking values": str(top5_vals).replace("'", ""),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def visualize_results(results):
    """Creates visualizations of the graph structure and ranking distribution."""
    for graph_name, graph_results in results.items():
        for d, d_results in graph_results.items():
            G = d_results["G"]
            pr = d_results["pr"]

            _, axes = plt.subplots(1, 2, figsize=(14, 6))

            pos = nx.kamada_kawai_layout(G)
            nx.draw_networkx(
                G,
                pos,
                ax=axes[0],
                node_size=[v * 3000 for v in pr],
                node_color=pr,
                cmap=plt.cm.viridis,
                with_labels=True,
                arrows=True,
            )
            axes[0].set_title(f"Random walk for {graph_name} (d={d})")
            axes[0].axis("off")

            axes[1].hist(pr, bins=30, color="skyblue", edgecolor="black")
            axes[1].set_title(f"Ranking histogram: {graph_name} (d={d})")
            axes[1].set_xlabel("Ranking value")
            axes[1].set_ylabel("Number of nodes")
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()
