from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from umap import UMAP

from emb3d import reader
from emb3d.types import VisualizeDisplayMode


def get_data(embedding_file: Path, title_field: str):
    with embedding_file.open() as fp:
        embeddings = []
        titles = []
        for idx, record in enumerate(reader.jsonl(fp)):
            embeddings.append(record["embedding"])
            titles.append(record.get(title_field, record.get("id", idx)))
        return pd.DataFrame(embeddings), titles


def umap_reduce(X):
    reducer = UMAP()
    return reducer.fit_transform(X)


def generate_chart(
    X_reduced,
    titles,
    kmeans,
    display_mode: VisualizeDisplayMode,
):
    if kmeans:
        cluster_labels = kmeans.labels_
        df_cluster_titles = pd.DataFrame({"title": titles, "cluster": cluster_labels})
        cluster_titles = df_cluster_titles.groupby("cluster").first()["title"]
        data = {
            "x1": X_reduced[:, 0],
            "x2": X_reduced[:, 1],
            "title": titles,
            "cluster": cluster_labels,
            "cluster_title": [cluster_titles[label] for label in cluster_labels],
        }
    else:
        data = {
            "x1": X_reduced[:, 0],
            "x2": X_reduced[:, 1],
            "title": titles,
        }

    df = pd.DataFrame(data)
    brush = alt.selection(type="interval")

    scatter_plot = (
        alt.Chart(df)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
            tooltip=["title", "cluster_title"] if kmeans else ["title"],
            color=alt.Color("cluster:N", legend=None) if kmeans else alt.value("blue"),
        )
        .add_selection(brush)
    )

    title_list = (
        alt.Chart(df)
        .mark_text()
        .encode(
            y=alt.Y("row_number:O", axis=None),
            text="title:N",
        )
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < 18)
        .properties(title="title")
    )

    if kmeans:
        if display_mode == VisualizeDisplayMode.CLUSTERS:
            centroids = kmeans.cluster_centers_
            chart = alt.Chart(
                pd.DataFrame(centroids, columns=["x1", "x2"])
            ).mark_circle(opacity=0.8, size=40, color="red")
        elif display_mode == VisualizeDisplayMode.RECORDS:
            chart = scatter_plot
        else:  # VisualizeDisplayMode.EVERYTHING
            centroids = kmeans.cluster_centers_
            centroids_chart = alt.Chart(
                pd.DataFrame(centroids, columns=["x1", "x2"])
            ).mark_circle(opacity=0.8, size=40, color="red")
            chart = scatter_plot + centroids_chart
    else:
        chart = scatter_plot

    chart = chart.interactive() | title_list

    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])
    return chart.add_selection(zoom)


def run_kmeans(X, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)


def get_cluster_titles(cluster_labels, titles, n_clusters):
    cluster_titles = []
    for i in range(n_clusters):
        cluster_idx = np.where(cluster_labels == i)[0]
        cluster_titles.append(titles[cluster_idx[0]])
    return cluster_titles
