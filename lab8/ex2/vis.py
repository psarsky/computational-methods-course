import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def display_results(results):
    rows = []
    for d, d_results in results.items():
        for name, result in d_results.items():
            top5 = [int(node) for node in result["top5"]]
            top5_vals = [f"{float(v):.3f}" for v in result["pr"][result["top5"]]]

            row = {
                "Damping factor": d,
                "Jump vector": name,
                "Iterations": result["iterations"],
                "Time": f"{result['time']:.6f}s",
                "Top 5 nodes": str(top5),
                "PageRank values": str(top5_vals).replace("'", ""),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def visualize_pagerank_comparison(G, results, d=0.85):
    jump_vectors = list(results[d].keys())
    num_vectors = len(jump_vectors)

    plt.figure(figsize=(4 * num_vectors, 6))
    pos = nx.spring_layout(G, seed=42)

    for i, name in enumerate(jump_vectors):
        plt.subplot(1, num_vectors, i + 1)
        pr = results[d][name]["pr"]
        nx.draw_networkx(
            G,
            pos,
            node_size=[v * 5000 for v in pr],
            node_color=pr,
            cmap=plt.cm.viridis,
            with_labels=True,
            arrows=True,
        )
        plt.title(f"PageRank with {name} (d={d})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pagerank_histogram(pr, title):
    plt.figure(figsize=(8, 4))
    plt.hist(pr, bins=50, color="skyblue", edgecolor="black")
    plt.title(f"PageRank histogram: {title}")
    plt.xlabel("PageRank value")
    plt.ylabel("Number of nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pagerank_histogram_large(pr, title):
    plt.figure(figsize=(10, 6))

    upper_limit = np.percentile(pr, 95)

    plt.hist(
        pr,
        bins=np.linspace(min(pr), upper_limit, 100),
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("PageRank value")
    plt.ylabel("Number of nodes")
    plt.title(f"PageRank histogram (Zoomed in): {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
