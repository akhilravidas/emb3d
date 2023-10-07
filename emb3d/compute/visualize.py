from pathlib import Path
from typing import Optional

import altair as alt
import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from umap import UMAP

from emb3d import reader
from emb3d.types import VisualizeDisplayMode

NUM_TITLES = 20


def run_hdbscan(X, min_cluster_size):
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(X)


def get_data(embedding_file: Path, title_field: Optional[str]):
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
    hdbscan_model,
):
    if hdbscan_model:
        cluster_labels = hdbscan_model.labels_
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
    brush = alt.selection_interval()

    scatter_plot = (
        alt.Chart(df)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
            tooltip=["title", "cluster_title"] if hdbscan_model else ["title"],
            color=alt.Color("cluster:N", legend=None)
            if hdbscan_model
            else alt.value("blue"),
        )
        .properties(width=1000)
        .add_params(brush)
    )

    title_list = (
        alt.Chart(df)
        .mark_text(align="left")
        .encode(
            y=alt.Y("row_number:O", axis=None),
            text="title:N",
        )
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < NUM_TITLES)
        .properties(title="Titles")
    )

    return (
        alt.vconcat(scatter_plot, title_list)
        .resolve_legend(color="independent")
        .configure_view(stroke=None)
    )


def run_kmeans(X, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
