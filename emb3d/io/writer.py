"""
Writers
"""
from pathlib import Path

import altair as alt


def chart2html(chart: alt.TopLevelMixin, out_fname: Path):
    """
    Write chart to html file
    """
    with alt.data_transformers.enable("default"):
        chart.save(out_fname)
