from typing import Optional, Tuple, Union

import genetools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_mean_and_standard_deviation_heatmap(
    data: pd.DataFrame,
    x_axis_key: str,
    y_axis_key: str,
    mean_key: str,
    standard_deviation_key: str,
    color_cmap: Optional[str] = None,
    color_vcenter: Optional[float] = None,
    figsize: Tuple[float, float] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot heatmap showing mean and standard deviation together.

    Circle color represents the mean. Circle size represents stability (inverse of standard deviation).

    In other words, big circles are trustworthy/stable across the average, while little circles aren't.
    And with a diverging colormap (e.g. `color_cmap='RdBu_r', color_vcenter=0`) bold circles are strong effects, while near-white circles are weak effects.
    """

    # Inspiration from:
    # https://stackoverflow.com/a/59384782/130164
    # https://stackoverflow.com/a/65654470/130164
    # https://stackoverflow.com/a/63559754/130164

    if figsize is None:
        # autosize
        n_cols = data[x_axis_key].nunique()
        n_rows = data[y_axis_key].nunique()
        figsize = (n_cols * 1.5, n_rows / 2.5)

    size_scaler = MinMaxScaler().fit(data[standard_deviation_key].values.reshape(-1, 1))
    min_marker_size = 20
    marker_size_scale_factor = 100

    def size_norm(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """convert raw size data to plot marker size"""
        return min_marker_size + marker_size_scale_factor * (
            1 - size_scaler.transform(np.array(x).reshape(-1, 1)).ravel()
        )

    def inverse_size_norm(converted_x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """convert plot marker size back to raw size data"""
        return size_scaler.inverse_transform(
            (
                1
                + min_marker_size / marker_size_scale_factor
                - np.array(converted_x) / marker_size_scale_factor
            ).reshape(-1, 1)
        ).ravel()

    size_data_normed = size_norm(data[standard_deviation_key])
    if not np.allclose(
        data[standard_deviation_key].values,
        inverse_size_norm(size_data_normed),
    ):
        raise ValueError("Inverting size norm did not result in original size")

    fig, ax = plt.subplots(figsize=figsize)

    try:
        scatter = ax.scatter(
            data[x_axis_key].values,
            data[y_axis_key].values,
            c=data[mean_key].values,
            s=size_data_normed,
            cmap=color_cmap,
            norm=CenteredNorm(vcenter=color_vcenter)
            if color_vcenter is not None
            else None,
            alpha=1,
        )
        ax.set_xlim(-0.5, max(ax.get_xticks()) + 0.5)
        ax.set_ylim(-0.5, max(ax.get_yticks()) + 0.5)
        ax.invert_yaxis()  # respect initial ordering - go top to bottom

        # Create grid
        ax.set_xticks(np.array(ax.get_xticks()) - 0.5, minor=True)
        ax.set_yticks(np.array(ax.get_yticks()) - 0.5, minor=True)
        ax.grid(which="minor")

        # Aspect ratio
        ax.set_aspect("equal", "box")

        # At this point, ax.get_xticklabels() may return empty tick labels and emit UserWarning: FixedFormatter should only be used together with FixedLocator
        # Must draw the canvas to position the ticks: https://stackoverflow.com/a/41124884/130164
        # And must assign tick locations prior to assigning tick labels, i.e. set_ticks(get_ticks()): https://stackoverflow.com/a/68794383/130164
        fig.canvas.draw()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation="vertical")

        # Make colorbar axes:
        # [xcorner, ycorner, width, height]: https://stackoverflow.com/a/65943132/130164
        cbar_ax = ax.inset_axes([1.1, 0.5, 0.3, 0.4], transform=ax.transAxes)

        # Make colorbar
        cbar = fig.colorbar(scatter, cax=cbar_ax).set_label(
            "Color (mean)",
            rotation=0,
            size="medium",
            weight="bold",
            horizontalalignment="left",
        )

        # Produce a legend with a cross section of sizes from the scatter.
        # "func" will inverse plot sizes back to original size
        handles, labels = scatter.legend_elements(
            prop="sizes",
            alpha=0.6,
            num=5,
            func=inverse_size_norm,
        )
        size_legend = ax.legend(
            handles,
            labels,
            bbox_to_anchor=(1.05, 0.25),
            loc="upper left",
            title="Size\n(inverse std. dev.)",
            borderaxespad=0.0,
            frameon=False,
            framealpha=0.0,
            title_fontproperties={"weight": "bold", "size": "medium"},
            numpoints=1,
            scatterpoints=1,
            markerscale=1.0,
        )
        # align legend title left
        size_legend._legend_box.align = "left"

        return fig, ax
    except Exception as err:
        # If there is an error, close the figure to prevent it from being displayed in a partial or broken state
        plt.close(fig)
        # Reraise
        raise err


def two_class_relative_density_plot(
    data,
    x_key,
    y_key,
    hue_key,
    positive_class,
    colorbar_label=None,
    quantile: Optional[float] = 0.50,
    figsize=(8, 8),
    n_bins=50,
    range=None,  # Extents within which to make bins
    continuous_cmap: str = "RdBu_r",
    cmap_vcenter: Optional[float] = 0.5,
    balanced_class_weights=True,
):
    """
    Two-class relative density plot.

    An earlier version is staged at https://github.com/maximz/genetools/pull/53/ - but this version has bugs fixed.
    """
    import scipy.stats

    def _weighted_mean(arr: np.ndarray, true_weight: float, false_weight: float):
        # how many total values in bin
        count = arr.shape[0]
        # how many positive class values in bin
        count_true = (arr.astype(int) == 1).sum()
        # how many negative class values in bin
        count_false = count - count_true

        numerator = count_true * true_weight
        return numerator / (numerator + count_false * false_weight)

    if balanced_class_weights:
        # Account for imbalance in positive and negative class sizes.
        # Members of rarer classes should count more towards density.
        # Instead of counting 1 towards density, each item should count 1/n_total_for_its_class.

        # Example: a bin with 2 positive and 2 negative examples.
        # But positive class has 1000 items overall and negative class has 10000 overall.
        # Unweighted mean: 2/(2+2) = 1/2.
        # Weighted: (2/1000) / [ (2/1000) + (2/10000) ] = 0.91.
        # The bin is significantly more positive than negative, relative to base rates.

        n_positive = (data[hue_key] == positive_class).sum()
        n_negative = data.shape[0] - n_positive
        statistic = lambda arr: _weighted_mean(
            arr=arr, true_weight=1 / n_positive, false_weight=1 / n_negative
        )
    else:
        statistic = "mean"

    binned_data = scipy.stats.binned_statistic_2d(
        data[x_key],
        data[y_key],
        data[hue_key] == positive_class,
        statistic=statistic,
        bins=n_bins,
        expand_binnumbers=True,
        range=range,
    )

    # which bin does each point belong to
    bin_number_df = pd.DataFrame(binned_data.binnumber, index=["x_bin", "y_bin"]).T

    # filter out any beyond-edge bins that capture values outside bin bounds (e.g. due to range parameter)
    # we don't want to modify `binned_data.statistic` for these bins, because their indices will be out of bounds in that array
    # and we don't want to include these bins in the bin sizes quantile calculation, since these bins won't be displayed on the plot.
    # (note that the bin numbers are 1-indexed, not 0-indexed! https://github.com/scipy/scipy/issues/7010#issuecomment-279264653)
    # (so anything in bin 0 or bin #bins+1 is out of bounds)
    bin_number_df = bin_number_df[
        (bin_number_df["x_bin"] >= 1)
        & (bin_number_df["x_bin"] <= n_bins)
        & (bin_number_df["y_bin"] >= 1)
        & (bin_number_df["y_bin"] <= n_bins)
    ]

    # bin sizes: number of points per bin
    bin_sizes = bin_number_df.groupby(["x_bin", "y_bin"]).size()

    # Fill N/A counts for bins with 0 items
    bin_sizes = bin_sizes.reindex(
        pd.MultiIndex.from_product(
            [
                np.arange(1, binned_data.statistic.shape[0] + 1),
                np.arange(1, binned_data.statistic.shape[1] + 1),
            ],
            names=bin_sizes.index.names,
        ),
        fill_value=0,
    )

    # Prepare to plot
    # binned_data.statistic does not follow Cartesian convention
    # we need to transpose to visualize.
    # see notes in numpy.histogram2d docs.
    plot_values = binned_data.statistic.T

    # Choose bins to remove: drop bins with low number of counts
    # i.e. low overall density
    if quantile is not None:
        bins_to_remove = bin_sizes[
            bin_sizes <= bin_sizes.quantile(quantile)
        ].reset_index(name="size")

        # Remove low-count bins by setting the color value to nan.
        # To multiple-index into a 2d array, list x dimensions first, then y dimensions second.
        # Note: the bin numbers are 1-indexed, not 0-indexed! (https://github.com/scipy/scipy/issues/7010#issuecomment-279264653)
        # Also note the swap of y and x in the bin numbers, because of the transpose above.
        # (See numpy.histogram2d docs for more info.)
        plot_values[
            bins_to_remove["y_bin"].values - 1, bins_to_remove["x_bin"].values - 1
        ] = np.nan

        remaining_bins = bin_sizes[bin_sizes > bin_sizes.quantile(quantile)]
    else:
        remaining_bins = bin_sizes

    # Plot, as in numpy histogram2d docs
    fig, ax = plt.subplots(figsize=figsize)
    pltX, pltY = np.meshgrid(binned_data.x_edge, binned_data.y_edge)
    colormesh = plt.pcolormesh(
        pltX,
        pltY,
        plot_values,
        cmap=continuous_cmap,
        norm=CenteredNorm(vcenter=cmap_vcenter) if cmap_vcenter is not None else None,
    )

    if colorbar_label is not None:
        # Add color bar.
        # see also https://stackoverflow.com/a/44642014/130164
        # Pull colorbar out of axis by creating a special axis for the colorbar - rather than distorting main ax.
        # specify width and height relative to parent bbox
        colorbar_ax = inset_axes(
            ax,
            width="5%",
            height="80%",
            loc="center left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        colorbar = fig.colorbar(colormesh, cax=colorbar_ax, label=colorbar_label)

        # set global "current axes" back to main axes,
        # so that any calls like plt.title target main ax rather than inset colorbar_ax
        plt.sca(ax)

    plt.xlabel(x_key)
    plt.ylabel(y_key)

    description = remaining_bins.describe()
    description = (
        f"Counts range from {description['min']:n} to {description['max']:n} per bin"
    )

    return fig, ax, description


def get_point_size(sample_size: int, maximum_size: float = 100) -> float:
    """get scatterplot point size based on sample size (from scanpy), but cut off at maximum_size"""
    # avoid division by zero - set sample_size to 1 if it's zero
    return min(120000 / max(1, sample_size), maximum_size)


def plot_confusion_matrix(
    df,
    figsize: Optional[Tuple[float, float]] = None,
    outside_borders=True,
    inside_border_width=0.5,
    wrap_labels_amount=15,
    wrap_x_axis_labels=True,
    wrap_y_axis_labels=True,
    draw_colorbar=False,
):
    # TODO: write test case with: df = pd.crosstab(pd.Series(['a', 'a', 'b', 'b', 'c']), pd.Series([1, 2, 3, 4, 1]))
    with sns.axes_style("white"):
        if figsize is None:
            # Automatic sizing of confusion matrix, based on df's shape
            margin = 0.25
            size_per_class = 0.8
            # width: give a little extra breathing room because horizontal labels will fill the space
            auto_width = margin * 2 + df.shape[1] * size_per_class * 1.2
            if not draw_colorbar:
                # remove some unnecessary width usually allocated to colorbar
                auto_width -= df.shape[1] / 5
            # height: don't need extra breathing room because labels go left-to-right not up-to-down
            auto_height = margin * 2 + df.shape[0] * size_per_class
            figsize = (auto_width, auto_height)
        fig, ax = plt.subplots(figsize=figsize)
        # add text with numeric values (annot=True), but without scientific notation (overriding fmt with "g" or "d")
        sns.heatmap(
            df,
            annot=True,
            fmt="g",
            cmap="Blues",
            ax=ax,
            linewidth=inside_border_width,
            cbar=draw_colorbar,
        )
        plt.setp(ax.get_yticklabels(), rotation="horizontal", va="center")
        plt.setp(ax.get_xticklabels(), rotation="horizontal")

        if outside_borders:
            # Activate outside borders
            for _, spine in ax.spines.items():
                spine.set_visible(True)

        if wrap_labels_amount is not None:
            # Wrap long tick labels
            genetools.plots.wrap_tick_labels(
                ax,
                wrap_amount=wrap_labels_amount,
                wrap_x_axis=wrap_x_axis_labels,
                wrap_y_axis=wrap_y_axis_labels,
            )

        return fig, ax
