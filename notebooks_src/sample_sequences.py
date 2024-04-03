# %% [markdown]
# # Sample sequences from our full ETL load
#
# - subselect sequences
# - subselect columns
# - remove specimens that violate some constraints: too few sequences, or not all isotypes found
#
# both peak + off-peak are still included after this.

# %%
from malid import config, helpers
from malid.sample_sequences import sample_sequences
import pandas as pd

# %%

# %%

# %% [markdown]
# **If regenerating, this notebook should automatically overwrite `config.paths.sequences_sampled`, but you can also manually clear it first with `rm -r`**

# %%
config.paths.sequences_sampled

# %%

# %%

# %%
import dask
import dask.dataframe as dd
import time

# %%

# %%

# %%

# %%
from dask.distributed import Client

# multi-processing backend
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

# %%
# Input columns
desired_columns = [
    "specimen_label",
    "participant_label",
    "amplification_label",
    "v_gene",
    "j_gene",
    "disease",
    "disease_subtype",
    "fr1_seq_aa_q_trim",
    "cdr1_seq_aa_q_trim",
    "fr2_seq_aa_q_trim",
    "cdr2_seq_aa_q_trim",
    "fr3_seq_aa_q_trim",
    "cdr3_seq_aa_q_trim",
    "post_seq_aa_q_trim",
    "cdr3_aa_sequence_trim_len",
    "extracted_isotype",
    "isotype_supergroup",
    "v_mut",
    "num_reads",
    "igh_or_tcrb_clone_id",
]

# %%
# Don't use fastparquet, because it changes specimen labels like M54-049 to 2049-01-01 00:00:54 -- i.e. it coerces partition names to numbers or dates
df = dd.read_parquet(config.paths.sequences, columns=desired_columns, engine="pyarrow")

# %%
# each partition is a specimen
df.npartitions

# %%
df

# %%
# required_gene_loci=config.gene_loci_used
# Required gene loci may differ for each specimen. Prepare a dict
required_gene_loci = helpers._load_etl_metadata()["available_gene_loci"]
required_gene_loci

# %%
required_gene_loci.value_counts()

# %%

# %%
# pass empty df as meta, along with the new columns created by sample_sequences
meta = df.head(0).assign(total_clone_num_reads=0, num_clone_members=0)
df_sampled = df.map_partitions(
    sample_sequences, required_gene_loci=required_gene_loci.to_dict(), meta=meta
)
df_sampled

# %%

# %%

# %%
config.paths.sequences_sampled

# %%
itime = time.time()

# This can behave weirdly with empty partitions. https://github.com/dask/dask/issues/8832

df_sampled.to_parquet(
    config.paths.sequences_sampled,
    overwrite=True,
    compression="snappy",  # gzip
    engine="pyarrow",
    # schema arg only accepted by pyarrow engine:
    # Set schema to "infer" if we have any empty partitions and using pyarrow.
    # schema="infer" is no longer slow as of https://github.com/dask/dask/pull/9131
    # schema=None breaks downstream readers.
    schema="infer",
    # also, do empty partitions even make it to disk, or are they eliminated? they seem eliminated.
    write_metadata_file=False,
    partition_on=["participant_label", "specimen_label"],
)

print(time.time() - itime)

# %%

# %%

# %%
df_sampled.dtypes

# %%
df_sampled2 = dd.read_parquet(config.paths.sequences_sampled, engine="pyarrow")
df_sampled2

# %%
# check dtypes
df_sampled2.dtypes

# %%
# compare dtypes
pd.concat(
    [
        df_sampled.dtypes.rename("expected dtypes"),
        df_sampled2.dtypes.rename("reloaded observed dtypes"),
    ],
    axis=1,
)

# %%
# expected lower because losing some empty specimens
df.npartitions, df_sampled.npartitions, df_sampled2.npartitions

# %%

# %%

# %%
client.shutdown()

# %%
