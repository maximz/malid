# %% [markdown]
# # Get all V genes and decide on an order of them. Also get all J genes.

# %%
import numpy as np
import matplotlib.pyplot as plt
import genetools
import seaborn as sns

sns.set_style("dark")

# %%
import pandas as pd

# %%
import dask
import dask.dataframe as dd

# %%
import os

# %%
from malid import config, helpers

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
desired_cols = ["v_gene", "j_gene", "isotype_supergroup"]

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
for gene_locus, isotype_groups in helpers.isotype_groups_kept.items():
    # Not sure why this doesn't work:
    # v_genes = df.loc[df["isotype_supergroup"].compute().isin(isotype_groups)]["v_gene"].unique().compute().sort_values()

    # Instead, here's a manual version using map_partitions:
    v_gene_unique_lists = df.map_partitions(
        lambda partdf: set(
            partdf[partdf["isotype_supergroup"].isin(isotype_groups)]["v_gene"].unique()
        )
    )
    j_gene_unique_lists = df.map_partitions(
        lambda partdf: set(
            partdf[partdf["isotype_supergroup"].isin(isotype_groups)]["j_gene"].unique()
        )
    )

    # compute
    v_gene_unique_lists, j_gene_unique_lists = dask.compute(
        v_gene_unique_lists, j_gene_unique_lists
    )

    # extract
    v_genes = pd.Series(
        list(set.union(*(v_gene_unique_lists.values))),
        name="v_gene",
    ).sort_values()
    j_genes = pd.Series(
        list(set.union(*(j_gene_unique_lists.values))),
        name="j_gene",
    ).sort_values()
    print(gene_locus, v_genes)
    print(gene_locus, j_genes)

    v_genes.to_csv(
        config.paths.dataset_specific_metadata
        / f"all_v_genes.in_order.{gene_locus.name}.txt",
        index=None,
    )
    j_genes.to_csv(
        config.paths.dataset_specific_metadata
        / f"all_j_genes.in_order.{gene_locus.name}.txt",
        index=None,
    )

# %%

# %%
client.shutdown()

# %%

# %%

# %%
