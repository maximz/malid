# Merge this back into genetools.
# Add screenshot tests.

import pytest
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from malid.external import genetools_plots


@pytest.fixture
def data():
    # Per numpy.histogram2d docs:
    # Generate non-symmetric test data
    n = 10000
    x = np.linspace(1, 100, n)
    y = 2 * np.log(x) + np.random.rand(n) - 0.5
    data = pd.DataFrame({"x": x, "y": y})
    return data


def test_scatter(data):
    ax = sns.scatterplot(data=data, x="x", y="y", alpha=0.5)
    return ax.get_figure()


def test_scatter_joint(data):
    g = sns.jointplot(data=data, x="x", y="y", alpha=0.5)
    return g.fig


def test_hexbin(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(
        data["x"],
        data["y"],
        gridsize=20,
        cmap="rainbow",
    )
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig


def test_overall_density(data):
    binned_data = scipy.stats.binned_statistic_2d(
        x=data["x"],
        y=data["y"],
        values=None,
        statistic="count",
        bins=20,
        expand_binnumbers=True,
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    pltX, pltY = np.meshgrid(binned_data.x_edge, binned_data.y_edge)
    plt.pcolormesh(pltX, pltY, binned_data.statistic.T, cmap="rainbow")
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig


def test_overall_density_filtered(data):
    n_bins = 20
    quantile = 0.50

    binned_data = scipy.stats.binned_statistic_2d(
        x=data["x"],
        y=data["y"],
        values=None,
        statistic="count",
        bins=n_bins,
        expand_binnumbers=True,
    )

    # which bin does each point belong to
    bin_number_df = pd.DataFrame(binned_data.binnumber, index=["x_bin", "y_bin"]).T

    # filter out any beyond-edge bins that capture values outside bin bounds (e.g. due to range parameter)
    bin_number_df = bin_number_df[
        (bin_number_df["x_bin"] >= 1)
        & (bin_number_df["x_bin"] <= n_bins)
        & (bin_number_df["y_bin"] >= 1)
        & (bin_number_df["y_bin"] <= n_bins)
    ]

    # bin sizes: number of points per bin
    bin_sizes = bin_number_df.groupby(["x_bin", "y_bin"], observed=False).size()

    # Fill N/A counts
    bin_sizes = bin_sizes.reindex(
        pd.MultiIndex.from_product(
            [
                range(1, binned_data.statistic.shape[0] + 1),
                range(1, binned_data.statistic.shape[1] + 1),
            ],
            names=bin_sizes.index.names,
        ),
        fill_value=0,
    )

    # choose bins to remove: drop bins with low number of counts, i.e. low overall density
    bins_to_remove = bin_sizes[bin_sizes <= bin_sizes.quantile(quantile)].reset_index(
        name="size"
    )

    # Plot
    # Note need to transpose and handle one-indexed bin IDs.
    newstat = binned_data.statistic.T
    newstat[
        bins_to_remove["y_bin"].values - 1, bins_to_remove["x_bin"].values - 1
    ] = np.nan
    fig, ax = plt.subplots(figsize=(5, 5))
    pltX, pltY = np.meshgrid(binned_data.x_edge, binned_data.y_edge)
    plt.pcolormesh(pltX, pltY, newstat, cmap="rainbow")
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig


def test_relative_density(data):
    data2 = pd.concat(
        [
            data.assign(classname="A"),
            pd.DataFrame(
                {
                    "x": np.random.randint(1, 100, 1000),
                    "y": np.random.randint(1, 10, 1000),
                    "classname": "B",
                }
            ),
        ],
        axis=0,
    )
    fig, ax, _ = genetools_plots.two_class_relative_density_plot(
        data2,
        x_key="x",
        y_key="y",
        hue_key="classname",
        positive_class="A",
        colorbar_label="proportion",
        quantile=0.90,
    )
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig
    # TODO: add test we have same results with balanced_class_weights=True or False when class frequencies are identical (e.g. bump B class to 10000).
    # Maybe return statistic directly so we can compare.
