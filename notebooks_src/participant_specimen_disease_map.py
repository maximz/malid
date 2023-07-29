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
    "disease",
    "disease_subtype",
    "specimen_time_point",
    "participant_age",
    "participant_description",
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
# df = df.drop_duplicates()
# display(df)
# dask.visualize(df)
# this has an aggregation step

# %%
# use map_partitions to avoid agg step that is unnecessary based on our partitioning strategy

metadata_df = df.map_partitions(lambda part: part.drop_duplicates())

# %%
metadata_df

# %%
# dask.visualize(metadata_df, filename="participant_specimen_metadata.dask_task_graph.pdf")

# %%
metadata_df_c = metadata_df.compute()

# %%
metadata_df_c

# %%
metadata_df_c = metadata_df_c.sort_values(["disease", "participant_label"])
metadata_df_c

# %%
# sanity check: one entry per participant + specimen
assert all(
    metadata_df_c.groupby(["participant_label", "specimen_label"], observed=True).size()
    == 1
)

# %%
metadata_df_c.to_csv(
    config.paths.dataset_specific_metadata / "participant_specimen_disease_map.tsv",
    sep="\t",
    index=None,
)

# %%
metadata_df_c = pd.read_csv(
    config.paths.dataset_specific_metadata / "participant_specimen_disease_map.tsv",
    sep="\t",
)

# %%

# %%
metadata_df_c["disease_subtype"].value_counts()

# %%

# %% [markdown]
# Confirm HIV patient numbers -- we expect:
#
# ```
# 43	HIV Negative
# 46	HIV Broad Neutralizing
# 50	HIV Non Neutralizing
# ```

# %%
# Specimens
metadata_df_c[metadata_df_c["disease"] == "HIV"]["disease_subtype"].astype(
    "category"
).cat.remove_unused_categories().value_counts()

# %%
# Patients
metadata_df_c[metadata_df_c["disease"] == "HIV"].groupby(
    "disease_subtype", observed=True
)["participant_label"].nunique()

# %%

# %%
# healthy specimens
metadata_df_c[metadata_df_c["disease"] == "Healthy/Background"][
    "disease_subtype"
].astype("category").cat.remove_unused_categories().value_counts()

# %%
# healthy patients
metadata_df_c[metadata_df_c["disease"] == "Healthy/Background"].groupby(
    "disease_subtype", observed=True
)["participant_label"].nunique()

# %%
metadata_df_c[metadata_df_c["disease_subtype"] == "Healthy/Background - CMV Unknown"]

# %%

# %%

# %%
# covid specimens
metadata_df_c[metadata_df_c["disease"] == "Covid19"]["disease_subtype"].astype(
    "category"
).cat.remove_unused_categories().value_counts()

# %%
# covid patients
metadata_df_c[metadata_df_c["disease"] == "Covid19"].groupby(
    "disease_subtype", observed=True
)["participant_label"].nunique()

# %%

# %%

# %%

# %%
client.shutdown()

# %%
