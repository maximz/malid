# %% [markdown]
# # Record which specimens remain after QC filtering in `sample_sequences.ipynb`

# %%
import numpy as np
import pandas as pd
from malid import config, helpers, logger

# %%
import dask
import dask.dataframe as dd

# %%

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
    memory_limit="125GB",  # per worker
)
display(client)
# for debugging: client.restart()

# %%
# Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
df = dd.read_parquet(config.paths.sequences_sampled, engine="pyarrow")
df

# %%
# each partition is a specimen
df.npartitions

# %%
df.columns

# %%

# %% [markdown]
# # Get all specimens available from ETL - meaning the ones that passed `sample_sequences` filters

# %%

# %%
# groupby participant, specimen, disease - get total sequence count
specimens = (
    df.groupby(
        ["participant_label", "specimen_label", "disease"],
        observed=True,
    )
    .size()
    .rename("total_sequence_count")
    .reset_index()
)
specimens

# %%

# %%
specimens = specimens.compute()
specimens

# %%
assert specimens.shape[0] == df.npartitions

# %%
assert not specimens["specimen_label"].duplicated().any()

# %%
# Export list of specimens remaining after QC filtering in sample_sequences.ipynb.
# Not all specimens survived to this step - some are thrown out for not having enough sequences or not having all isotypes.
# However, these aren't yet filtered to is_selected_for_cv_strategy specimens that are particular to the selected cross validation strategy.
specimens.to_csv(
    config.paths.dataset_specific_metadata
    / "specimens_that_survived_qc_filters_in_sample_sequences_notebook.tsv",
    sep="\t",
    index=None,
)

# %%
