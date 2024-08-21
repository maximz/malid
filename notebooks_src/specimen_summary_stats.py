# -*- coding: utf-8 -*-
# %% [markdown]
# # Summary statistics

# %%

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
from malid import helpers

# %%
import genetools

# %%
from malid.datamodels import GeneLocus

# %%
from malid import io

# %%

# %% [markdown]
# ### Load

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
# if already opened from another notebook, see https://stackoverflow.com/questions/60115736/dask-how-to-connect-to-running-cluster-scheduler-and-access-total-occupancy
client = Client(
    scheduler_port=config.dask_scheduler_port,
    dashboard_address=config.dask_dashboard_address,
    n_workers=config.dask_n_workers,
    processes=True,
    threads_per_worker=8,
    memory_limit="auto",
    worker_dashboard_address=":0",  # start worker dashboards on random ports
)
display(client)
# for debugging: client.restart()

# %%

# %%
# Filter to specimens that are kept in the training set and have all gene loci available
metadata = helpers.get_all_specimen_info()
specimen_labels = list(
    metadata[
        (metadata["in_training_set"])
        & (metadata["available_gene_loci"] == config.gene_loci_used)
    ]["specimen_label"].unique()
)
filters = [("specimen_label", "in", list(specimen_labels))]
len(specimen_labels)

# %%

# %%
desired_cols = [
    "specimen_label",
    "disease",
    "isotype_supergroup",
    "cdr3_aa_sequence_trim_len",
    "v_mut",
    "igh_or_tcrb_clone_id",
]

# %%
# Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
df = dd.read_parquet(
    config.paths.sequences_sampled,
    columns=desired_cols,
    filters=filters,
    engine="pyarrow",
)
df

# %%

# %%
# Mark which sequences are BCR vs TCR
df["gene_locus"] = (df["isotype_supergroup"] == "TCRB").map(
    {True: GeneLocus.TCR.name, False: GeneLocus.BCR.name}
)
df

# %%

# %% [markdown]
# ### Calculate

# %%
# %%time
# Coverage: Number of clones per sample
# (BCR: count each clone once, even if we keep several copies of it in different isotypes)
num_clones_per_sample = df.map_partitions(
    lambda part: part.groupby(
        ["specimen_label", "disease", "gene_locus"],
        observed=True,
    )["igh_or_tcrb_clone_id"]
    .nunique()
    .rename("num_clones_per_specimen")
).compute()
num_clones_per_sample.head()

# %%
# One entry per specimen x gene locus
num_clones_per_sample.shape

# %%

# %%

# %% [markdown]
# ## Plot

# %%
with sns.axes_style("white"), sns.plotting_context("paper"):
    ax = sns.boxplot(
        data=num_clones_per_sample.reset_index(),
        x="disease",
        hue="gene_locus",
        y="num_clones_per_specimen",
        flierprops=dict(
            markerfacecolor="0.75",
            markersize=5,
            linestyle="none",
            #
            marker="x",
            markeredgecolor="0.75",
        ),
        order=list(sorted(num_clones_per_sample.reset_index()["disease"].unique())),
        palette=sns.color_palette("Paired")[:2],
    )
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            ax.get_xticklabels(),
            # Don't double count specimens across gene loci
            num_clones_per_sample.reset_index().groupby("specimen_label").first(),
            "disease",
        )
    )
    genetools.plots.wrap_tick_labels(ax, wrap_amount=10, wrap_y_axis=False)
    sns.despine(ax=ax)
    sns.move_legend(
        ax, "center left", bbox_to_anchor=(1, 0.5), title=None, frameon=False
    )
    plt.xlabel("Disease")
    plt.ylabel("Number of clones\nper sample", rotation=0, ha="right")
    genetools.plots.savefig(
        ax.get_figure(),
        config.paths.base_output_dir_for_selected_cross_validation_strategy
        / "repertoire_summary_stats.num_clones_per_specimen_by_disease.png",
        dpi=300,
    )

# %%

# %%

# %%

# %%
# Example of how the raw data looks:

# %%
io.load_raw_parquet_sequences_for_specimens(
    specimen_labels=[specimen_labels[0]],
    gene_locus=GeneLocus.BCR,
    fname=config.paths.sequences,
).iloc[0].to_dict()

# %%
io.load_raw_parquet_sequences_for_specimens(
    specimen_labels=[specimen_labels[0]],
    gene_locus=GeneLocus.TCR,
    fname=config.paths.sequences,
).iloc[0].to_dict()

# %%

# %% [markdown]
# # Also look at pre-sampling data

# %%
desired_cols = [
    "isotype_supergroup",
    "trimmed_sequence",
]

# %%
# Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
df = dd.read_parquet(
    config.paths.sequences,  # not sequences_sampled
    columns=desired_cols,
    filters=filters,
    engine="pyarrow",
)
df

# %%
# Mark which sequences are BCR vs TCR
df["gene_locus"] = (df["isotype_supergroup"] == "TCRB").map(
    {True: GeneLocus.TCR.name, False: GeneLocus.BCR.name}
)
df

# %%
# %%time
# Nucleotide length
df["sequence_length"] = df["trimmed_sequence"].str.len()
sequence_lengths = df.groupby(["gene_locus"])["sequence_length"].mean().compute()
sequence_lengths

# %%
df.groupby(["gene_locus"])["sequence_length"].median().compute()

# %%

# %%

# %%
# Clean up
client.shutdown()

# %%
