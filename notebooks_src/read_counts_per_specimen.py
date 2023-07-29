# -*- coding: utf-8 -*-
# %% [markdown]
# # Get read counts per specimen. Using parquet files

# %%
import numpy as np
import matplotlib.pyplot as plt
import genetools
import seaborn as sns

sns.set_style("dark")

# %%
import pandas as pd

# %%
from slugify import slugify

# %%
import dask
import dask.dataframe as dd

# %%
import os

# %%
from malid import config

# %%

# %% [markdown]
# Raise worker connection timeouts (see other dask notebooks):

# %%
import distributed

# These only seem to be picked up by scheduler, not by individual workers

dask.config.set(
    {
        "distributed.comm.timeouts.tcp": "120s",
        "distributed.comm.timeouts.connect": "120s",
        "distributed.comm.retry.count": 5,
    }
)

# %%
# These will be picked up by individual workers

with open(os.path.expandvars("$HOME/.config/dask/distributed.yaml"), "w") as w:
    w.write(
        """distributed:
  comm:
    retry:
      count: 5
    timeouts:
      connect: 120s          # time before connecting fails
      tcp: 120s              # time before calling an unresponsive connection dead
    """
    )

# %%
from dask.distributed import Client

# multi-processing backend
# access dashbaord at http://127.0.0.1:61083
# if already opened from another notebook, see https://stackoverflow.com/questions/60115736/dask-how-to-connect-to-running-cluster-scheduler-and-access-total-occupancy
client = Client(
    scheduler_port=61084,
    dashboard_address=":61083",
    n_workers=7,
    processes=True,
    threads_per_worker=8,
    memory_limit="auto",
    worker_dashboard_address=":0",  # start worker dashboards on random ports
)
display(client)
# for debugging: client.restart()

# %%
desired_cols = [
    "participant_label",
    "specimen_label",
    "extracted_isotype",
    "disease",
    "disease_subtype",
    "num_reads",
]

# %%
debug_filters = None
# debug_filters = [("participant_label", "==", "BFI-0007450")]

# %%
# Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
df = dd.read_parquet(
    config.paths.sequences,
    columns=desired_cols,
    filters=debug_filters,
    engine="pyarrow",
)

# %%
df

# %%

# %%
# sum num_reads and also sum of unique vdj sequences (count)

# %%
num_reads_per_specimen_isotype = df.map_partitions(
    lambda part: part.groupby(
        [
            "participant_label",
            "specimen_label",
            "disease",
            "disease_subtype",
            "extracted_isotype",
        ],
        observed=True,
    )["num_reads"]
    .sum()
    .rename("num_reads_per_specimen_isotype")
)

# %%
num_unique_vdj_per_specimen_isotype = df.map_partitions(
    lambda part: part.groupby(
        [
            "participant_label",
            "specimen_label",
            "disease",
            "disease_subtype",
            "extracted_isotype",
        ],
        observed=True,
    )
    .size()
    .rename("num_unique_vdj_per_specimen_isotype")
)

# %%
# dask.visualize(
#     num_reads_per_specimen_isotype,
#     num_unique_vdj_per_specimen_isotype,
#     filename="read_counts_per_specimen.dask_task_graph.pdf",
# )

# %%
# %%time
num_reads_per_specimen_isotype_c, num_unique_vdj_per_specimen_isotype_c = dask.compute(
    num_reads_per_specimen_isotype, num_unique_vdj_per_specimen_isotype
)

# %%
num_reads_per_specimen_isotype_c

# %%
num_unique_vdj_per_specimen_isotype_c

# %%

# %%
num_reads_per_specimen_isotype_df = num_reads_per_specimen_isotype_c.reset_index()
num_reads_per_specimen_isotype_df.head()

# %%
num_unique_vdj_per_specimen_isotype_df = (
    num_unique_vdj_per_specimen_isotype_c.reset_index()
)
num_unique_vdj_per_specimen_isotype_df.head()

# %%

# %%
# Clean up
client.shutdown()

# %%

# %% [markdown]
# # Export pivoted

# %%
num_reads_per_specimen_isotype_df_pivot = num_reads_per_specimen_isotype_df.pivot(
    index=["participant_label", "specimen_label", "disease", "disease_subtype"],
    columns=["extracted_isotype"],
    values="num_reads_per_specimen_isotype",
)
num_reads_per_specimen_isotype_df_pivot

# %%
num_reads_per_specimen_isotype_df_pivot.to_csv(
    f"{config.paths.base_output_dir}/reads_per_specimen_and_isotype.tsv", sep="\t"
)

# %%

# %%
num_unique_vdj_per_specimen_isotype_df_pivot = (
    num_unique_vdj_per_specimen_isotype_df.pivot(
        index=["participant_label", "specimen_label", "disease", "disease_subtype"],
        columns=["extracted_isotype"],
        values="num_unique_vdj_per_specimen_isotype",
    )
)
num_unique_vdj_per_specimen_isotype_df_pivot

# %%
num_unique_vdj_per_specimen_isotype_df_pivot.to_csv(
    f"{config.paths.base_output_dir}/unique_vdj_per_specimen_and_isotype.tsv", sep="\t"
)

# %%

# %%

# %% [markdown]
# # Plots of all specimens

# %%
for disease, grp in num_unique_vdj_per_specimen_isotype_df.groupby("disease"):
    # Remove missing categories so they are not plotted below as empty boxes
    grp = grp.copy()
    grp["specimen_label"] = grp["specimen_label"].cat.remove_unused_categories()

    g = sns.catplot(
        data=grp,
        x="extracted_isotype",
        y="num_unique_vdj_per_specimen_isotype",
        col="specimen_label",
        col_wrap=5,
        kind="bar",
        sharey=True,
        #     aspect=0.6,
        sharex=True,
        ci=False,
    )

    # Remove "specimen_label =" prefix for titles
    g.set_titles("{col_name}")

    g.set_ylabels(
        "# unique VDJ sequences\nin specimen", rotation=0, horizontalalignment="right"
    )

    # repeat x axis labels for all facets
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    # Clear titles
    g.set_titles("")

    # Set title with other dataframe columns
    # can't change title with .map() because .map() will set titles again
    for ax, ((i, j, k), data) in zip(g.axes.flat, g.facet_data()):
        # for each facet:
        # i,j,k are indexes into row_, col_, hue_names attributes
        # data is the subsetted data for this facet
        # could also get ax by: ax = fg.facet_axis(i, j)
        if data.empty:
            # hide plot entirely if it's empty
            ax.set_axis_off()
        else:
            ax.set_title(
                f'{data["specimen_label"].iloc[0]} — {data["disease_subtype"].iloc[0]}'
            )

    genetools.plots.savefig(
        g.fig,
        f"{config.paths.base_output_dir}/repertoire_read_counts.unique_vdj_sequences.{slugify(disease)}.png",
        dpi=72,
    )

# %%

# %%
for disease, grp in num_reads_per_specimen_isotype_df.groupby("disease"):
    # Remove missing categories so they are not plotted below as empty boxes
    grp = grp.copy()
    grp["specimen_label"] = grp["specimen_label"].cat.remove_unused_categories()

    g = sns.catplot(
        data=grp,
        x="extracted_isotype",
        y="num_reads_per_specimen_isotype",
        col="specimen_label",
        col_wrap=5,
        kind="bar",
        sharey=True,
        #     aspect=0.6,
        sharex=True,
        ci=False,
    )

    # Remove "specimen_label =" prefix for titles
    g.set_titles("{col_name}")

    g.set_ylabels("Total reads\nin specimen", rotation=0, horizontalalignment="right")

    # repeat x axis labels for all facets
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    # Clear titles
    g.set_titles("")

    # Set title with other dataframe columns
    # can't change title with .map() because .map() will set titles again
    for ax, ((i, j, k), data) in zip(g.axes.flat, g.facet_data()):
        # for each facet:
        # i,j,k are indexes into row_, col_, hue_names attributes
        # data is the subsetted data for this facet
        # could also get ax by: ax = fg.facet_axis(i, j)
        if data.empty:
            # hide plot entirely if it's empty
            ax.set_axis_off()
        else:
            ax.set_title(
                f'{data["specimen_label"].iloc[0]} - {data["disease_subtype"].iloc[0]}'
            )

    genetools.plots.savefig(
        g.fig,
        f"{config.paths.base_output_dir}/repertoire_read_counts.total_reads.{slugify(disease)}.png",
        dpi=72,
    )

# %%

# %% [markdown]
# # Roll up by disease

# %%
g = sns.catplot(
    data=num_reads_per_specimen_isotype_df.groupby(["disease", "extracted_isotype"])[
        "num_reads_per_specimen_isotype"
    ]
    .median()
    .reset_index(),
    x="extracted_isotype",
    y="num_reads_per_specimen_isotype",
    col="disease",
    col_wrap=5,
    kind="bar",
    sharey=True,
    #     aspect=0.6,
    sharex=True,
    ci=False,
)
g.set_ylabels(
    "Median total reads\nper specimen", rotation=0, horizontalalignment="right"
)


genetools.plots.savefig(
    g.fig,
    f"{config.paths.base_output_dir}/repertoire_read_counts.total_reads.by_disease.shared_ylims.png",
    dpi=100,
)

# %%
g = sns.catplot(
    data=num_reads_per_specimen_isotype_df.groupby(["disease", "extracted_isotype"])[
        "num_reads_per_specimen_isotype"
    ]
    .median()
    .reset_index(),
    x="extracted_isotype",
    y="num_reads_per_specimen_isotype",
    col="disease",
    col_wrap=5,
    kind="bar",
    sharey=False,
    #     aspect=0.6,
    sharex=True,
    ci=False,
)
g.set_ylabels(
    "Median total reads\nper specimen", rotation=0, horizontalalignment="right"
)

genetools.plots.savefig(
    g.fig,
    f"{config.paths.base_output_dir}/repertoire_read_counts.total_reads.by_disease.png",
    dpi=100,
)

# %%

# %%
g = sns.catplot(
    data=num_unique_vdj_per_specimen_isotype_df.groupby(
        ["disease", "extracted_isotype"]
    )["num_unique_vdj_per_specimen_isotype"]
    .median()
    .reset_index(),
    x="extracted_isotype",
    y="num_unique_vdj_per_specimen_isotype",
    col="disease",
    col_wrap=5,
    kind="bar",
    sharey=True,
    #     aspect=0.6,
    sharex=True,
    ci=False,
)
g.set_ylabels(
    "Median unique VDJ\nsequences per specimen", rotation=0, horizontalalignment="right"
)

genetools.plots.savefig(
    g.fig,
    f"{config.paths.base_output_dir}/repertoire_read_counts.unique_vdj_sequences.by_disesase.shared_ylims.png",
    dpi=100,
)

# %%
g = sns.catplot(
    data=num_unique_vdj_per_specimen_isotype_df.groupby(
        ["disease", "extracted_isotype"]
    )["num_unique_vdj_per_specimen_isotype"]
    .median()
    .reset_index(),
    x="extracted_isotype",
    y="num_unique_vdj_per_specimen_isotype",
    col="disease",
    col_wrap=5,
    kind="bar",
    sharey=False,
    #     aspect=0.6,
    sharex=True,
    ci=False,
)
g.set_ylabels(
    "Median unique VDJ\nsequences per specimen", rotation=0, horizontalalignment="right"
)


genetools.plots.savefig(
    g.fig,
    f"{config.paths.base_output_dir}/repertoire_read_counts.unique_vdj_sequences.by_disesase.png",
    dpi=100,
)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # Graveyard (figuring out how to plot this)

# %%
g = sns.catplot(
    data=num_reads_per_specimen_isotype_df.groupby(["disease", "extracted_isotype"])[
        "num_reads_per_specimen_isotype"
    ]
    .median()
    .reset_index(),
    x="disease",
    y="num_reads_per_specimen_isotype",
    col="extracted_isotype",
    col_wrap=5,
    kind="bar",
    #     aspect=0.6,
    sharex=True,
    ci=False,
)

# %%

# %%
test_df = pd.DataFrame(
    {
        "participant_label": ["a", "a", "b", "c"] * 2,
        "specimen_label": [1, 2, 3, 4] * 2,
        "extracted_isotype": ["IgA"] * 4 + ["IgD"] * 4,
        "num_reads_per_specimen_isotype": [10, 6, 8, 9, 5, 15, 4, 3],
    }
)
test_df

# %%
# num_reads_per_specimen_isotype_c
g = sns.catplot(
    data=test_df,
    x="extracted_isotype",
    y="num_reads_per_specimen_isotype",
    row="participant_label",
    col="specimen_label",
    kind="bar",
    sharey=True,
)

# %%
# num_reads_per_specimen_isotype_c
g = sns.catplot(
    data=test_df,
    x="extracted_isotype",
    y="num_reads_per_specimen_isotype",
    col="specimen_label",
    col_wrap=3,
    kind="bar",
    sharey=True,
    aspect=0.6,
    sharex=True,
)
# repeat x axis labels for all facets
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)

# %%

# %%
test_df = pd.DataFrame(
    {
        "specimen_label": [1, 2, 3, 4, 5, 6, 7, 8] * 2,
        "extracted_isotype": ["IgA"] * 4 + ["IgD"] * 4 + ["IgE"] * 4 + ["IgM"] * 4,
        "num_reads_per_specimen_isotype": [10, 6, 8, 9, 5, 15, 4, 3] * 2,
    }
)
test_df

# %%
# num_reads_per_specimen_isotype_c
g = sns.catplot(
    data=test_df,
    x="extracted_isotype",
    y="num_reads_per_specimen_isotype",
    col="specimen_label",
    col_wrap=6,
    kind="bar",
    sharey=True,
    aspect=0.6,
    sharex=True,
)

# Remove "specimen_label =" prefix for titles
g.set_titles("{col_name}")

g.set_ylabels("Total reads\nin specimen", rotation=0, horizontalalignment="right")

# repeat x axis labels for all facets
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)

# %%

# %%
# num_unique_vdj_per_specimen_isotype = (
#     df.groupby(["participant_label", "specimen_label"])
#     .size()
#     .rename("num_unique_vdj_per_specimen_isotype")
# )

# %%
# num_unique_vdj_per_specimen_isotype

# %%
# dask.visualize(
#     num_unique_vdj_per_specimen_isotype,
#     filename="num_unique_vdj_per_specimen_isotype.dask_task_graph.pdf",
# )

# %%
# num_unique_vdj_per_specimen_isotype_c = dask.compute(
#     num_unique_vdj_per_specimen_isotype
# )

# %%
# num_unique_vdj_per_specimen_isotype_c

# %%
# num_unique_vdj_per_specimen_isotype_c = num_unique_vdj_per_specimen_isotype_c[0]

# %%
# num_unique_vdj_per_specimen_isotype_c.max()

# %%
# num_unique_vdj_per_specimen_isotype_c = (
#     num_unique_vdj_per_specimen_isotype_c.reset_index()
# )

# %%
# # test
# num_unique_vdj_per_specimen_isotype_c["extracted_isotype"] = "IgA"

# %%
# g = sns.catplot(
#     data=num_unique_vdj_per_specimen_isotype_c,
#     x="extracted_isotype",
#     y="num_unique_vdj_per_specimen_isotype",
#     col="specimen_label",
#     col_wrap=6,
#     kind="bar",
#     sharey=True,
#     aspect=0.6,
#     sharex=True,
#     ci=False,
# )

# # Remove "specimen_label =" prefix for titles
# g.set_titles("{col_name}")

# g.set_ylabels("Total reads\nin specimen", rotation=0, horizontalalignment="right")

# # repeat x axis labels for all facets
# for ax in g.axes.flatten():
#     ax.tick_params(labelbottom=True)

# %%
