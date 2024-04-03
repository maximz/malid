# %% [markdown]
# # ETL
#
# **See dask worker logs on disk.**
#
# Convert csvs to parquet. The resulting parquet files are partitioned by `participant_label` and `specimen_label`, so we can run `df.map_partitions(lambda part: ...)` to execute a function on each specimen.
#
# The parquet dataset will include all datasets: in-house and Adaptive together.

# %%

# %%

# %%

# %%
import os
import pandas as pd
import glob
import time
import dask, dask.distributed
import dask.dataframe as dd
from dask.distributed import Client
from IPython.display import display
from typing import Dict
from malid import config
from malid.datamodels import GeneLocus
from malid.etl import (
    dtypes_expected_after_preprocessing,
    preprocess_each_participant_table,
    load_participant_data_external,
    read_boydlab_participant_table,
)

# %%
config.paths.sequences

# %%
# multi-processing backend
# if already opened from another notebook, see https://stackoverflow.com/questions/60115736/dask-how-to-connect-to-running-cluster-scheduler-and-access-total-occupancy
client = Client(
    scheduler_port=config.dask_scheduler_port,
    dashboard_address=config.dask_dashboard_address,
    n_workers=config.dask_n_workers,
    processes=True,
    threads_per_worker=8,
    # memory_limit="auto",
    # Still experimenting with this:
    memory_limit=0,  # no limit
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


# %%
@dask.delayed
def load_participant(files: Dict[GeneLocus, str], metadata_whitelist: pd.DataFrame):
    df_parts = []
    for gene_locus, fname in files.items():
        df_parts.append(
            preprocess_each_participant_table(
                df=read_boydlab_participant_table(fname, gene_locus),
                gene_locus=gene_locus,
                metadata_whitelist=metadata_whitelist,
            )
        )

    # combine BCR + TCR data from same participant.
    # necessary because we output one parquet partition per specimen - including both loci.
    # note that any or all parts may be empty dataframes (with .shape[0] == 0), but that's ok, as long as the columns and dtypes are correct.
    return pd.concat(df_parts, axis=0).reset_index(drop=True)


# %%

# %%
bcr_directories_to_read = [
    # NOTE: Some of the HHCs have been renamed as ".bz2.bak" in hhc_bcr_part_tables,
    # because they were resequenced later and had part tables reexported in other run directories (e.g. M477/M482).
    f"{config.paths.base_data_dir}/hhc_bcr_part_tables/part_table_*.bz2",
    #
    f"{config.paths.base_data_dir}/hiv_bcr_part_tables/part_table_*.bz2",
    f"{config.paths.base_data_dir}/covid19_buffycoat/bcr/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M418_M434_Covid_SamYang/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M447_M448_pediatric_lupus/BCR_M447/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M454_M455_adult_lupus_rna/BCR_M454/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M456_M457_adult_lupus_paxgene/BCR_M456/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M464_M463_healthy_children/BCR_M465/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M477_M482_yoni_ibd_and_some_old_hhc/BCR_M477/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M479_M484_gubatan_ibd_and_some_old_hhc/BCR_M479/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M433_M435_UPENN_Influenza_Study_2021/BCR_M433_M435/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M491_M493_diabetes_biobank/BCR_M491_M492/part_table_*.bz2",
    # These datasets are BCR only:
    f"{config.paths.base_data_dir}/covid19_seattle/part_table_*.bz2",
    f"{config.paths.base_data_dir}/lupus_m281redo/part_table_*.bz2",
]
tcr_directories_to_read = [
    # NOTE: Some of the HHCs have been renamed as ".bz2.bak" in hhc_tcr_part_tables,
    # because they were resequenced later and had part tables reexported in other run directories.
    f"{config.paths.base_data_dir}/hhc_tcr_part_tables/part_table_*.bz2",
    #
    f"{config.paths.base_data_dir}/hiv_tcr_part_tables/part_table_*.bz2",
    f"{config.paths.base_data_dir}/covid19_buffycoat/tcr/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M419_Covid_SamYang_tcrb/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M447_M448_pediatric_lupus/TCR_M448/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M454_M455_adult_lupus_rna/TCR_M455/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M456_M457_adult_lupus_paxgene/TCR_M457/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M464_M463_healthy_children/TCR_M463/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M477_M482_yoni_ibd_and_some_old_hhc/TCR_M482/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M479_M484_gubatan_ibd_and_some_old_hhc/TCR_M484/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M433_M435_UPENN_Influenza_Study_2021/TCR_M444/part_table_*.bz2",
    f"{config.paths.base_data_dir}/M491_M493_diabetes_biobank/TCR_M493/part_table_*.bz2",
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
), "Some expected part tables are missing: " + str(
    set(specimen_whitelist_and_metadata["fname"]) - set(files_trimmed["fname_trim"])
)

# %%

# %%
files_trimmed["fname_trim"].nunique(), files_trimmed.shape[0]

# %%

# %%
# Load Adaptive metadata
adaptive_metadata = pd.read_csv(
    config.paths.metadata_dir / "adaptive" / "generated.adaptive_external_cohorts.tsv",
    sep="\t",
)
adaptive_metadata

# %%
# Load other external cohort metadata
other_external_metadata = pd.read_csv(
    config.paths.metadata_dir / "generated.external_cohorts.tsv",
    sep="\t",
)
other_external_metadata

# %%

# %%
# all Delayed() objects
part_tables = []

# in-house data
for key, grp in files_trimmed.groupby("fname_trim"):
    # We have now selected all files for this participant, because fname_trim is something like part_table_BFI-#######.bz2 (there's a BCR file with that name and a TCR file with that name).
    # The participant is spread out over several rows in files_trimmed by locus and by specimen - even though ultimately there is one source file on disk per locus per participant.

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

# Adaptive data (TCR)
delayed_load_func = dask.delayed(load_participant_data_external)
for key, grp in adaptive_metadata.groupby("participant_label"):
    part_tables.append(
        delayed_load_func(
            participant_samples=grp,
            gene_locus=GeneLocus.TCR,
            base_path=config.paths.external_raw_data / "adaptive_immuneaccess",
            is_adaptive=True,
        )
    )

# Other external data (BCR or TCR)
for key, grp in other_external_metadata.groupby("participant_label"):
    gene_locus = grp["gene_locus"].unique()
    # Allow some studies to be exempted from read count column requirements
    expect_a_read_count_column = grp["expect_a_read_count_column"].unique()
    # Allow custom file extensions for some studies. Default is tsv
    file_extension = grp["file_extension"].unique()

    # All participants are either BCR or TCR, not both
    assert len(gene_locus) == 1
    gene_locus = gene_locus[0]

    # Other columns should also have single value
    assert len(expect_a_read_count_column) == 1
    expect_a_read_count_column = expect_a_read_count_column[0]
    assert len(file_extension) == 1
    file_extension = file_extension[0]

    part_tables.append(
        delayed_load_func(
            participant_samples=grp,
            # convert back from name attribute to full GeneLocus object
            gene_locus=GeneLocus[gene_locus],
            # Under this main directory are /study_name folders that include the samples and the parsed.IgH.tsv or parsed.TCRB.tsv files
            base_path=config.paths.external_raw_data,
            is_adaptive=False,
            expect_a_read_count_column=expect_a_read_count_column,
            file_extension=file_extension,
        )
    )

# Later, consider giving the dask.delayed objects custom names, e.g. the participant label as name, so we can identify them in dashboard and track down errors.
# see https://docs.dask.org/en/latest/delayed-api.html#dask.delayed.delayed and https://docs.dask.org/en/stable/custom-collections.html#implementing-deterministic-hashing
# Perhaps this will also enable us to rearrange the order in which jobs are run. We had tried to randomly shuffle the part_tables list, but from_delayed seemed to ignore our shuffling. Are the jobs run in the order of their (currently random) dask.delayed name attributes?
#
# Easiest way to do this may be:
# delayed_task = delayed_load_func(...)
# delayed_task.key = f"{delayed_task.key}_{gene_locus}_{grp['participant_label'].iloc[0]}"
# part_tables.append(delayed_task)
# This appens specific information to the task's existing key, which Dask has already made sure is unique. Therefore we get unique identifier plus our own custom information.
#
# Alternative:
# delayed_task = delayed(myfunc_not_yet_wrapped, dask_key_name=f"load_data_{gene_locus}_{grp['participant_label'].iloc[0]}")(...)
# part_tables.append(delayed_task)
# Here we directly control the task's key, but it's on us to make sure keys are unique and never reused.
#
# (We've tested neither approach ourselves.)

df = dd.from_delayed(
    part_tables, meta=dtypes_expected_after_preprocessing, verify_meta=False
)

# %%
df

# %%

# %%
config.paths.sequences

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

# %%
df2 = dd.read_parquet(config.paths.sequences, engine="pyarrow")

# %%
# check dtypes
df2

# %%
# compare dtypes
pd.concat(
    [
        df.dtypes.rename("expected dtypes"),
        df2.dtypes.rename("reloaded observed dtypes"),
    ],
    axis=1,
)

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
