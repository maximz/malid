# %% [markdown]
# # Get dotplot data
#
# For each clone, get:
#
# - V gene (identical for all clone members)
# - CDR3 length (identical for all clone members)
# - V-region mutation level, median across unique VDJ sequences in the clone (unweighted by read counts)
# - Total read count, summed across all unique VDJ sequences in the clone

# %%

# %%

# %%
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import genetools
import seaborn as sns
from dask.distributed import Client

sns.set_style("dark")
import os
import time
from malid import config, helpers

# %%
client = Client(
    scheduler_port=config.dask_scheduler_port,
    dashboard_address=config.dask_dashboard_address,
    processes=True,
    n_workers=config.dask_n_workers,
    threads_per_worker=8,
    memory_limit="auto",  # TOTAL_MEMORY * min(1, nthreads / total_nthreads)
    worker_dashboard_address=":0",  # start worker dashboards on random ports
    local_directory="/tmp",  # https://github.com/dask/distributed/issues/3559#issuecomment-597321459
)

display(client)
# for debugging: client.restart()

# %%

# %%
helpers.get_all_specimen_info()

# %%
desired_columns = [
    "extracted_isotype",
    "specimen_label",
    "participant_label",
    "v_gene",
    #     "disease",
    #     "disease_subtype",
    "igh_or_tcrb_clone_id",
    "v_mut",
    "num_reads",
    "cdr3_aa_sequence_trim_len",
]

# %%
# debug_filters = [
#     ("specimen_label", "==", helpers.get_all_specimen_info().iloc[0]["specimen_label"])
# ]

# %%
# Each partition is one specimen
df = dd.read_parquet(
    config.paths.sequences,
    columns=desired_columns,
    #     filters=debug_filters, # Re-enable to debug
    engine="pyarrow",
)

# %%
df

# %%

# %%
grpcols = [
    "participant_label",
    "specimen_label",
    "extracted_isotype",
    "igh_or_tcrb_clone_id",
]

# %%
# colors
clone_v_mut_median = df.map_partitions(
    lambda part: part.groupby(grpcols, observed=True)["v_mut"]
    .apply(pd.Series.median)
    .rename("clone_v_mut_median")
)
clone_v_mut_median

# %%
# reads per clone
sizes = df.map_partitions(
    lambda part: part.groupby(grpcols, observed=True)["num_reads"]
    .sum()
    .rename("clone_size")
)
sizes

# %%
vgene = df.map_partitions(
    lambda part: part.groupby(grpcols, observed=True)["v_gene"].first()
)
vgene

# %%
cdr3len = df.map_partitions(
    lambda part: part.groupby(grpcols, observed=True)[
        "cdr3_aa_sequence_trim_len"
    ].first()
)
cdr3len

# %%

# %%
# dask.visualize(
#     clone_v_mut_median, sizes, vgene, cdr3len, filename="/tmp/dask.png"
# )  # filename=None has a bug

# %%
# %%time
clone_v_mut_median_c, sizes_c, vgene_c, cdr3len_c = dask.compute(
    clone_v_mut_median, sizes, vgene, cdr3len
)

# %%

# %%
clone_v_mut_median_c

# %%
sizes_c

# %%
vgene_c

# %%
cdr3len_c

# %%

# %%

# %%
merged_df = pd.concat([clone_v_mut_median_c, sizes_c, vgene_c, cdr3len_c], axis=1)
merged_df

# %%
merged_df = merged_df.reset_index()

# %%
merged_df

# %%
merged_df["extracted_isotype"].value_counts()

# %%
assert not merged_df["extracted_isotype"].isna().any()

# %%

# %%
# Export by participant label

# %%
config.paths.dotplots_input

# %%
for participant_label, participant_grp in merged_df[
    merged_df["extracted_isotype"] != "gDNA"
].groupby("participant_label", observed=True):
    fname_out = config.paths.dotplots_input / f"{participant_label}.tsv"  # .gz
    participant_grp.to_csv(fname_out, sep="\t", index=None)

# %%

# %%

# %%
