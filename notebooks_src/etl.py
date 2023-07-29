# %% [markdown]
# # ETL
#
# **See dask worker logs on disk.**
#
# Convert csvs to parquet. The resulting parquet files are partitioned by `participant_label` and `specimen_label`, so we can run `df.map_partitions(lambda part: ...)` to execute a function on each specimen.

# %%

# %%

# %%

# %%
import os
import pandas as pd
import glob
import time
import dask
import dask.dataframe as dd
from IPython.display import display
from typing import Dict
from malid import config
from malid.datamodels import GeneLocus
from malid.etl import (
    dtypes_read_in,
    dtypes_expected_after_preprocessing,
    preprocess_each_participant_table,
    fix_dtypes,
)

# %%
config.paths.sequences

# %%

# %%
from dask.distributed import Client

dask.config.set({"logging.distributed": "info"})

# multi-processing backend
# access dashbaord at http://127.0.0.1:61083
client = Client(
    scheduler_port=61084,
    dashboard_address=":61083",
    n_workers=8,  # 4
    processes=True,
    threads_per_worker=8,
    memory_limit="auto",  # "125GB" per worker
    local_directory="/tmp",
)


def setup_worker_logging(dask_worker: dask.distributed.worker.Worker):
    import malid
    from notebooklog import setup_logger

    malid.logger, log_fname = setup_logger(
        log_dir=config.paths.log_dir, name=f"dask_worker_{dask_worker.name}"
    )
    malid.logger.info(log_fname)
    print(log_fname)


# Setup logging to disk on every current and future worker
# https://stackoverflow.com/questions/41475239/how-to-set-up-logging-on-dask-distributed-workers
client.register_worker_callbacks(setup=setup_worker_logging)

display(client)
# for debugging: client.restart()

# %%


# %%
cols = {
    GeneLocus.BCR: list(dtypes_read_in[GeneLocus.BCR].keys()),
    GeneLocus.TCR: list(dtypes_read_in[GeneLocus.TCR].keys()),
}

# %% [markdown]
# If we try to do `df = dd.read_csv(fnames, sep="\t", compression="bz2", dtype=dtypes, usecols=cols)`, it works but with:
#
# ```
# /home/maxim/miniconda3/lib/python3.7/site-packages/dask/dataframe/io/csv.py:459: UserWarning: Warning bz2 compression does not support breaking apart files
# Please ensure that each individual file can fit in memory and
# use the keyword ``blocksize=None to remove this message``
# Setting ``blocksize=None``
#   "Setting ``blocksize=None``" % compression
# ```

# %%
# df = dd.read_csv(fnames, sep="\t", compression="bz2", dtype=dtypes, usecols=cols)

# %%

# %%
# manual load with special processing:
# deduping and setting num_reads, setting extracted_isotype, setting disease and disease_subtype


# %%
allowed_hiv_runs = ["M111", "M112", "M113", "M114", "M124", "M125", "M132"]


# %%
@dask.delayed
def load_participant(files: Dict[GeneLocus, str], metadata_whitelist: pd.DataFrame):
    final_dtypes = dtypes_expected_after_preprocessing  # not dependent on locus
    df_parts = []
    for gene_locus, fname in files.items():
        df_for_locus = pd.read_csv(
            fname, sep="\t", dtype=dtypes_read_in[gene_locus], usecols=cols[gene_locus]
        )

        # filter out anything except whitelisted specimens
        # this means df.shape[0] can become 0
        df_for_locus = pd.merge(
            df_for_locus,
            metadata_whitelist,
            how="inner",
            on=["participant_label", "specimen_label"],
        )

        if df_for_locus.shape[0] == 0:
            # empty sample at this point - skip rest of processing this locus
            continue

        # override some variables
        df_for_locus["participant_label"] = df_for_locus[
            "participant_label_override"
        ].fillna(df_for_locus["participant_label"])
        df_for_locus["specimen_time_point"] = df_for_locus[
            "specimen_time_point_override"
        ].fillna(df_for_locus["specimen_time_point"])

        # if this is a patient from the HIV cohort: allow specimens from certain runs only
        if (
            df_for_locus.shape[0] > 0 and df_for_locus["hiv_run_filter"].iloc[0] == True
        ):  # must check shape[0] > 0 so iloc[0] does not fail
            # select certain run IDs only. exclude very old runs (M52 and such)
            # this means df.shape[0] can become 0
            df_for_locus = df_for_locus.loc[
                df_for_locus["run_label"].isin(allowed_hiv_runs)
            ]

        df_parts.append(
            preprocess_each_participant_table(
                df=df_for_locus.reset_index(drop=True),
                gene_locus=gene_locus,
                final_dtypes=final_dtypes,
            )
        )

    # combine BCR + TCR data from same participant. necessary because we output one parquet partition per specimen - including both loci
    if len(df_parts) == 0:
        # return empty dataframe but with the right columns + dtypes
        return fix_dtypes(pd.DataFrame(), final_dtypes)

    return pd.concat(df_parts, axis=0).reset_index(drop=True)


# %%

# %%
bcr_directories_to_read = [
    f"{config.paths.base_data_dir}/hhc_bcr_part_tables/part_table_*.bz2",
    f"{config.paths.base_data_dir}/hiv_bcr_part_tables/part_table_*.bz2",
    f"{config.paths.base_data_dir}/covid19_buffycoat/bcr/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M418_M434_Covid_SamYang/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M447_M448_pediatric_lupus/BCR_M447/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M454_M455_adult_lupus_rna/BCR_M454/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M456_M457_adult_lupus_paxgene/BCR_M456/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M464_M463_healthy_children/BCR_M465/part_table_*.bz2",
    # These datasets are BCR only:
    f"{config.paths.base_data_dir}/covid19_seattle/part_table_*.bz2",
    f"{config.paths.base_data_dir}/lupus_m281redo/part_table_*.bz2",
]
tcr_directories_to_read = [
    f"{config.paths.base_data_dir}/hhc_tcr_part_tables/part_table_*.bz2",
    f"{config.paths.base_data_dir}/hiv_tcr_part_tables/part_table_*.bz2",
    f"{config.paths.base_data_dir}/covid19_buffycoat/tcr/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M419_Covid_SamYang_tcrb/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M447_M448_pediatric_lupus/TCR_M448/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M454_M455_adult_lupus_rna/TCR_M455/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M456_M457_adult_lupus_paxgene/TCR_M457/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M464_M463_healthy_children/TCR_M463/part_table_*.bz2",
]

# %%
dfs = []
for gene_locus, locus_dirs in zip(
    [GeneLocus.BCR, GeneLocus.TCR], [bcr_directories_to_read, tcr_directories_to_read]
):
    for dirname in locus_dirs:
        fnames = list(glob.glob(dirname))
        if len(fnames) == 0:
            # The path must be wrong
            raise ValueError(f"No part tables found in {dirname} for {gene_locus}")
        dfs.append(pd.DataFrame({"fname_full": fnames, "gene_locus": gene_locus.name}))

files = pd.concat(dfs, axis=0).reset_index(drop=True)
files["fname_trim"] = files["fname_full"].apply(os.path.basename)
files.shape

# %%

# %%
files.head()

# %%
# # debug only:
# # files = files.iloc[-10:]
# files = files.sort_values("fname_trim").iloc[:4]
# files

# %%

# %%
# TODO: switch to helpers._load_etl_metadata()
specimen_whitelist_and_metadata = pd.read_csv(
    f"{config.paths.metadata_dir}/generated_combined_specimen_metadata.tsv",
    sep="\t",
)
specimen_whitelist_and_metadata

# %%

# %%
# filter to matching participant labels, so we're not loading part tables only to throw them out completely
# we might still throw them out partially (some specimens)
assert not specimen_whitelist_and_metadata["participant_label"].isna().any()
specimen_whitelist_and_metadata["fname"] = (
    "part_table_" + specimen_whitelist_and_metadata["participant_label"] + ".bz2"
)
specimen_whitelist_and_metadata["fname"]

# %%
specimen_whitelist_and_metadata["fname"].nunique()


# %%
files_trimmed = pd.merge(
    files,  # left side will have one row per locus per participant
    specimen_whitelist_and_metadata,  # right side will have one row per specimen per participant
    left_on="fname_trim",
    right_on="fname",
    how="inner",
)

assert (
    files_trimmed["fname_trim"].nunique()
    == specimen_whitelist_and_metadata["fname"].nunique()
), "Some expected part tables are missing"

# %%

# %%
files_trimmed["fname_trim"].nunique(), files_trimmed.shape[0]

# %%

# %%
# all Delayed() objects
part_tables = []

for key, grp in files_trimmed.groupby("fname_trim"):
    # We have now selected all files for this participant
    # Spread out over several rows by locus and by specimen - even though ultimately there is one source file on disk per locus per participant
    # Drop specimen dupes:
    unique_locus_files_for_this_participant = (
        grp[["fname_full", "gene_locus"]]
        .drop_duplicates()
        .set_index("gene_locus")["fname_full"]
    )
    if unique_locus_files_for_this_participant.index.duplicated().any():
        raise ValueError(
            "Multiple unique files on disk for the same locus for the same participant - should be one file per locus per participant"
        )
    part_tables.append(
        load_participant(
            files={
                GeneLocus[locus_name]: fname
                for locus_name, fname in unique_locus_files_for_this_participant.to_dict().items()
            },
            metadata_whitelist=specimen_whitelist_and_metadata,
        )
    )

df = dd.from_delayed(
    part_tables, meta=dtypes_expected_after_preprocessing, verify_meta=False
)

# %%
df

# %%
itime = time.time()

# %%
# This can behave weirdly with empty partitions: https://github.com/dask/dask/issues/8832 - requires being careful with engine, schema, and metadata

# fastparquet engine seems buggy, perhaps due to empty parititons too:
# OverflowError: value too large to convert to int
# Exception ignored in: 'fastparquet.cencoding.write_thrift'
# Traceback (most recent call last):
#   File "/users/maximz/anaconda3/envs/cuda-env-py39/lib/python3.9/site-packages/fastparquet/writer.py", line 1488, in write_thrift
#     return f.write(obj.to_bytes())

df.to_parquet(
    config.paths.sequences,
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

# %%
etime = time.time()

# %%
etime - itime

# %%

# %%
df2 = dd.read_parquet(config.paths.sequences, engine="pyarrow")

# %%
# check dtypes
df2

# %%
df2.dtypes

# %%
df.dtypes

# %%
# expected higher because now divided by participant_label and specimen_label
df.npartitions, df2.npartitions

# %%

# %%
# df2 = dd.read_parquet(config.paths.sequences, engine="fastparquet")

# %% [markdown]
# This warning `Partition names coerce to values of different types, e.g. ['M64-079', Timestamp('2039-01-01 00:00:54')]` is a serious problem for us; we need to avoid `fastparquet` as a result.

# %%
# # check dtypes
# df2

# %%
# df2.dtypes

# %%
# df.dtypes

# %%

# %%

# %%
client.shutdown()

# %%
