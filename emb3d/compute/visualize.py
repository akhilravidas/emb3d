from pathlib import Path
from typing import Optional

import altair as alt
import hdbscan
import numpy as np
import pandas as pd
from umap import UMAP

from emb3d.io import reader

NUM_TITLES = 20
READ_CHUNK_SIZE = 500


def cluster_hdbscan(X: pd.DataFrame, min_cluster_size: int) -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(X)


# TODO: Very very inefficient, time and heap allocation wise
def get_data(embedding_file: Path, label_field: Optional[str]):
    embeddings = []
    titles = []

    chunk_iter = pd.read_json(embedding_file, lines=True, chunksize=READ_CHUNK_SIZE)

    offset = 0
    for chunk in chunk_iter:
        embeddings.extend(chunk["embedding"].tolist())
        titles.extend(
            chunk.get(label_field, chunk.get("id", chunk.index + offset)).tolist()
        )
        offset += len(chunk)

    df_embeddings = pd.DataFrame(embeddings)

    return df_embeddings, titles


def umap_reduce(X: pd.DataFrame) -> np.ndarray:
    reducer = UMAP()
    return reducer.fit_transform(X)  # type: ignore


def generate_chart(X_reduced, titles, hdbscan_model) -> alt.TopLevelMixin:
    if hdbscan_model:
        cluster_labels = hdbscan_model.labels_
        df_cluster_titles = pd.DataFrame(
            {
                "x1": X_reduced[:, 0],
                "x2": X_reduced[:, 1],
                "title": titles,
                "cluster": cluster_labels,
            }
        )

        df_agg = (
            df_cluster_titles.groupby("cluster")
            .agg(
                x1=("x1", "mean"),
                x2=("x2", "mean"),
                count=("cluster", "size"),
                cluster_title=("title", "first"),
            )
            .reset_index()
        )

        data = df_agg
    else:
        data = {
            "x1": X_reduced[:, 0],
            "x2": X_reduced[:, 1],
            "title": titles,
        }
        data = pd.DataFrame(data)

    brush = alt.selection_interval()

    scatter_plot = (
        alt.Chart(data)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
            tooltip=["cluster_title", "count"] if hdbscan_model else ["title"],
            color=alt.Color("cluster:N", legend=None)
            if hdbscan_model
            else alt.value("blue"),
            size=alt.Size("count:Q", legend=None, scale=alt.Scale(range=[10, 200]))
            if hdbscan_model
            else alt.value(20),
        )
        .properties(width=1000)
        .add_params(brush)
    )

    title_list = (
        alt.Chart(data)
        .mark_text(align="left")
        .encode(
            y=alt.Y("row_number:O", axis=None),
            text="cluster_title:N" if hdbscan_model else "title:N",
        )
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < NUM_TITLES)
    )

    return (
        alt.vconcat(scatter_plot, title_list)
        .resolve_legend(color="independent")
        .configure_view(stroke=None)
    )
