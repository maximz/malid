# -*- coding: utf-8 -*-
# %% [markdown]
# # Create clone IDs for each patient.
#
# Create clone IDs across all specimens (i.e. all timepoints, mixing peak and non-peak samples -- and including all replicates of those timepoints) from a patient.
#
# Clone IDs are not unique across patients

# %%
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

# %%
from malid import config, helpers, etl, get_v_sequence, io, logger
from malid.datamodels import GeneLocus, healthy_label
from malid.sample_sequences import sample_sequences
from malid.trained_model_wrappers import ConvergentClusterClassifier

# %%

# %%
n_jobs = 40

# %%

# %% [markdown]
# # get specimen filepaths from specimen metadata list

# %%

# %% [markdown]
# ## covid samples

# %%
covid_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid19_bcr.specimen_metadata_extra.tsv",
    sep="\t",
)
covid_specimens

# %%
covid_specimens.shape

# %%
participant_df = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid19_bcr.participant_metadata.tsv",
    sep="\t",
)
participant_df

# %%
covid_specimens = pd.merge(
    covid_specimens, participant_df, how="left", validate="m:1", on="participant_label"
)
covid_specimens

# %%
covid_specimens.shape

# %%
covid_specimens["disease_subtype"] = (
    covid_specimens["disease"]
    + " - "
    + covid_specimens["study_name"]
    + covid_specimens["is_peak"].replace({True: "", False: " (non-peak)"})
)
covid_specimens["gene_locus"] = GeneLocus.BCR.name
covid_specimens

# %%
covid_specimens["fname"] = covid_specimens.apply(
    lambda row: config.paths.external_raw_data
    / "covid_external_as_part_tables"
    / f"exported.part_table.{row['specimen_label']}.tsv",
    axis=1,
)
covid_specimens.head()

# %%

# %% [markdown]
# ## healthy specimens

# %%
healthy_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.healthy_bcr.participant_metadata.tsv",
    sep="\t",
)

# process peak samples only
healthy_specimens = healthy_specimens[healthy_specimens["is_peak"] == True]

healthy_specimens["disease_subtype"] = (
    healthy_specimens["disease"]
    + " - "
    + healthy_specimens["study_name"]
    + healthy_specimens["is_peak"].replace({True: "", False: " (non-peak)"})
)

healthy_specimens["gene_locus"] = GeneLocus.BCR.name

healthy_specimens

# %%
healthy_specimens["fname"] = healthy_specimens.apply(
    lambda row: config.paths.external_raw_data
    / "briney_healthy_as_part_tables"
    / f"exported.part_table.{row['specimen_label']}.tsv",
    axis=1,
)
healthy_specimens.head()

# %%

# %% [markdown]
# ## healthy TCR specimens

# %%
tcr_healthy_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.healthy_tcr_britanova.participant_metadata.tsv",
    sep="\t",
).assign(
    is_peak=True,
    gene_locus=GeneLocus.TCR.name,
)
tcr_healthy_specimens["fname"] = tcr_healthy_specimens["specimen_label"].apply(
    lambda specimen_label: config.paths.external_raw_data
    / "chudakov_aging"
    / f"{specimen_label}.txt.gz"
)

tcr_healthy_specimens

# %%

# %% [markdown]
# ## Covid TCR specimens

# %%
tcr_covid_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.covid_tcr_shomuradova.participant_metadata.tsv",
    sep="\t",
).assign(
    is_peak=True,
    gene_locus=GeneLocus.TCR.name,
)
tcr_covid_specimens["fname"] = tcr_covid_specimens["specimen_label"].apply(
    lambda specimen_label: config.paths.external_raw_data
    / "shomuradova"
    / f"split.{specimen_label}.tsv"
)

tcr_covid_specimens

# %% [markdown]
# ## Adaptive healthy TCR specimens

# %%
adaptive_tcr_healthy_specimens = pd.DataFrame(
    {"fname": (config.paths.external_raw_data / "emerson").glob("*.tsv")}
).assign(
    disease=healthy_label,
    study_name="Emerson",
    is_peak=True,
    gene_locus=GeneLocus.TCR.name,
    # Flag that these are a different platform than what we expect
    different_platform=True,
)

# extract specimen label from filename
adaptive_tcr_healthy_specimens = adaptive_tcr_healthy_specimens.assign(
    specimen_label=adaptive_tcr_healthy_specimens["fname"].apply(lambda path: path.stem)
)
assert not adaptive_tcr_healthy_specimens["specimen_label"].duplicated().any()
# participants are 1:1 with specimens
adaptive_tcr_healthy_specimens["participant_label"] = adaptive_tcr_healthy_specimens[
    "specimen_label"
]

# TODO: add sex, age, ethnicity

adaptive_tcr_healthy_specimens["disease_subtype"] = (
    adaptive_tcr_healthy_specimens["disease"]
    + " - "
    + adaptive_tcr_healthy_specimens["study_name"]
    + adaptive_tcr_healthy_specimens["is_peak"].replace(
        {True: "", False: " (non-peak)"}
    )
)

adaptive_tcr_healthy_specimens

# %%

# %% [markdown]
# ## Adaptive Covid TCR specimens

# %%
adaptive_tcr_covid_specimens = pd.read_csv(
    config.paths.metadata_dir
    / "generated.external_cohorts.adaptive_covid_tcr.specimens.tsv",
    sep="\t",
).assign(
    study_name="ImmuneCode",
    is_peak=True,
    gene_locus=GeneLocus.TCR.name,
    # Flag that these are a different platform than what we expect
    different_platform=True,
)
adaptive_tcr_covid_specimens["fname"] = adaptive_tcr_covid_specimens[
    "specimen_label"
].apply(
    lambda specimen_label: config.paths.external_raw_data
    / "immunecode"
    / "reps"
    / "ImmuneCODE-Review-002"
    / f"{specimen_label}.tsv"
)

adaptive_tcr_covid_specimens

# %%

# %% [markdown]
# ## Instructions for adding more
#
# If we have external cohorts that are BCR+TCR, we should have one row per locus per specimen.
#
# Set a `specimen_label_by_locus` column that is the globally-unique specimen label tailored to a particular locus, e.g. `$SPECIMENLABEL-IGH` or `$SPECIMENLABEL-TRB` format.
#
# And set a `specimen_label` column that is equivalent across different-loci rows for that specimen.
#
# The row's `gene_locus` column should be set to the locus of that row (must be the name of a valid `GeneLocus` enum value), and the `fname` column should be set to the path to the file containing the data for that locus.

# %%

# %% [markdown]
# ## merge

# %%
all_specimens = pd.concat(
    [
        covid_specimens,
        healthy_specimens,
        tcr_healthy_specimens,
        tcr_covid_specimens,
        adaptive_tcr_healthy_specimens,
        adaptive_tcr_covid_specimens,
    ],
    axis=0,
)

# fillna
all_specimens["different_platform"].fillna(False, inplace=True)

all_specimens

# %%
# Fillna for cohorts that are single-locus
if "specimen_label_by_locus" not in all_specimens:
    # in case we had no BCR+TCR combined cohorts that set this field already
    all_specimens["specimen_label_by_locus"] = all_specimens["specimen_label"]
else:
    all_specimens["specimen_label_by_locus"].fillna(
        all_specimens["specimen_label"], inplace=True
    )

# %%
# make sure input fnames exist
assert all_specimens["fname"].apply(os.path.exists).all()
# %%
all_specimens.shape

# %%

# %%
# Set age_group column as well, just as in assemble_etl_metadata
all_specimens["age"].describe()

# %%
all_specimens["age_group"] = pd.cut(
    all_specimens["age"],
    bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
    labels=["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"],
    right=False,
)
all_specimens["age_group"].value_counts()

# %%
all_specimens["age_group"].cat.categories

# %%
all_specimens["age"].isna().value_counts()

# %%
all_specimens["age_group"].isna().value_counts()

# %%
for age_group, grp in all_specimens.groupby("age_group"):
    print(age_group, grp["age"].min(), grp["age"].max())

# %%
# Just as in assemble_etl_metadata:
# Null out "age_group" column for extreme ages with small sample size.

# Note that we are not getting rid of these specimens altogether,
# but marking age_group NaN will disable their use for demographics-controlling models

orig_shapes = all_specimens.shape[0], all_specimens["age_group"].isna().sum()
mask = all_specimens["age_group"].isin(["80+"])
all_specimens.loc[mask, "age_group"] = np.nan
new_shapes = all_specimens.shape[0], all_specimens["age_group"].isna().sum()

# sanity checks:
# - we did not drop any specimens
assert orig_shapes[0] == new_shapes[0]
# - but we did null out some age_group entries
assert orig_shapes[1] < new_shapes[1]
# - we nulled out the right amount
assert new_shapes[1] - orig_shapes[1] == mask.sum()

# %%

# %%
# export for later processing
all_specimens.drop(["fname"], axis=1).to_csv(
    config.paths.metadata_dir / "generated.external_cohorts.all_specimens.tsv",
    sep="\t",
    index=None,
)

# %%

# %%
# confirm all specimen labels are unique within each locus (may have one BCR and one TCR line per specimen)
# TODO: in the future, allow for replicates of each specimen
assert not all_specimens["specimen_label_by_locus"].duplicated().any()
for locus, grp in all_specimens.groupby("gene_locus"):
    assert not grp["specimen_label"].duplicated().any()

# %%
# Which specimens are in multiple loci?
all_specimens[all_specimens["specimen_label"].duplicated(keep=False)]

# %%
all_specimens["study_name"].value_counts()

# %%
all_specimens["disease"].value_counts()

# %%
all_specimens["gene_locus"].value_counts()

# %%
all_specimens["disease_subtype"].value_counts()

# %%
all_specimens["different_platform"].value_counts()

# %%
all_specimens.groupby(["different_platform", "disease_subtype"]).size()

# %%

# %%
for demographics_column in ["age", "age_group", "sex", "ethnicity_condensed"]:
    print(demographics_column)
    print(all_specimens[demographics_column].value_counts())
    print(all_specimens[demographics_column].isna().value_counts())
    print()


# %%

# %%
# TODO: Separate all above into a separate "compile all metadata" notebook.

# %%

# %%

# %% [markdown]
# # process specimen, drop duplicates, and cluster each group (-> clones)
#

# %%
def process_specimen(
    fname: Path, gene_locus: GeneLocus, study_name: str, specimen_label: str
):
    # defensive cast
    fname = Path(fname)

    # each specimen is one "repertoire_id"
    df = pd.read_csv(
        fname,
        sep="\t",
        # Solve "DtypeWarning: Columns (9,17,25,33) have mixed types. Specify dtype option on import or set low_memory=False."
        dtype={
            "pre_seq_nt_q": "object",
            "pre_seq_nt_v": "object",
            "pre_seq_nt_d": "object",
            "pre_seq_nt_j": "object",
        },
        # special N/A values for Adaptive data
        na_values=["no data", "unknown", "unresolved"],
    )

    # Rename columns
    if "specimen_label" not in df.columns:
        if "sample_name" in df.columns:
            df.rename(
                columns={
                    "sample_name": "specimen_label",
                },
                inplace=True,
            )
        elif "repertoire_id" in df.columns:
            df.rename(
                columns={
                    "repertoire_id": "specimen_label",
                },
                inplace=True,
            )
        else:
            df = df.assign(specimen_label=specimen_label)

    # confirm only one specimen included here
    if not (df["specimen_label"] == specimen_label).all():
        raise ValueError(
            f"Processing specimen {specimen_label}, but specimen_label column in {fname} does not match this."
        )

    # Recognize sample type
    if study_name == "Britanova" and gene_locus == GeneLocus.TCR:
        # Special case for this study:
        # No raw sequences available to run IgBlast ourselves.

        # Rename columns
        df.rename(
            columns={
                "v": "v_gene",
                "j": "j_gene",
                "cdr3nt": "cdr3_seq_nt_q",
                "cdr3aa": "cdr3_seq_aa_q",
                "count": "num_reads",
            },
            inplace=True,
        )

        # Pulling in CDR1 and CDR2 for TCRB data in etl._compute_columns expects v_segment to be set with allele info.
        # etl._compute_columns guarantees that each V gene has a CDR1 and CDR2 available for its dominant *01 allele.
        df["v_segment"] = df["v_gene"] + "*01"
        df["j_segment"] = df["j_gene"] + "*01"

        # Trim CDR3 AA: remove ends
        # and replace field that's entirely space (or empty) with NaN
        # (maybe we should be trimming cdr3-nt too, but that's only used for clustering within patient to set clone IDs, so it doesn't matter)
        df.dropna(subset=["cdr3_seq_aa_q"], inplace=True)
        df["cdr3_seq_aa_q"] = (
            df["cdr3_seq_aa_q"]
            .str.slice(start=1, stop=-1)
            .replace(r"^\s*$", np.nan, regex=True)
        )
        df.dropna(subset=["cdr3_seq_aa_q"], inplace=True)

        # Mark these all as productive TCRB
        df["productive"] = True
        df["extracted_isotype"] = "TCRB"

    else:
        # We have run sequences through our IgBlast.
        # Note that our legacy IgBlast parser uses the location of our Boydlab primers to parse the CDR3 sequence
        # So if you run shorter sequences like Adaptive seuqences, we won't parse IgBlast's CDR3 calls, but we will still get V/J calls.
        is_adaptive = "bio_identity" in df.columns

        if study_name in ["Briney", "Kim", "Montague"]:
            # Special case: iReceptor / VDJserver / Briney studies that went through legacy Postgres-based annotations,
            # i.e. igblast parses already merged in.

            # TODO: update runbook for these to be consistent with new schema where we merge igblast parses on the fly,
            # then eliminate this special case.

            # Igblast parse exported from Postgres database: cast to bool to be compatible
            df["productive"] = df["productive"].replace({"t": True, "f": False})

            # get v_sequence (same way we produce v_sequence in internal pipeline's sort script)
            (
                df["v_sequence"],
                df["d_sequence"],
                df["j_sequence"],
            ) = get_v_sequence.complete_sequences(df)

        else:
            if is_adaptive:
                # Adaptive, reprocessed through our IgBlast
                # See https://www.adaptivebiotech.com/wp-content/uploads/2019/07/MRK-00342_immunoSEQ_TechNote_DataExport_WEB_REV.pdf

                # Our IgBlast gives some different V gene calls, but generally doesn't provide CDR3 calls for these short sequences.
                # That's because our parser looks for the location of our primers.
                # We'll use the V/J gene and productive calls from our IgBlast, while using Adaptive's CDR3 call.

                if "cdr3_rearrangement" in df.columns:
                    df.rename(
                        columns={"cdr3_rearrangement": "cdr3_seq_nt_q"}, inplace=True
                    )
                elif "rearrangement" in df.columns:
                    # this is the entire detected rearrangement, not just cdr3, but will suffice for finding nucleotide uniques
                    df.rename(columns={"rearrangement": "cdr3_seq_nt_q"}, inplace=True)
                else:
                    raise ValueError(
                        f"No nucleotide rearrangement column found for {specimen_label}"
                    )

                # Define whether productive from Adaptive metadata
                df["productive"] = df["frame_type"] == "In"

                # Count copies using templates or seq_reads field, whichever is available
                if not df["templates"].isna().any():
                    df["num_reads"] = df["templates"]
                elif not df["seq_reads"].isna().any():
                    df["num_reads"] = df["seq_reads"]
                else:
                    raise ValueError(
                        f"Could not choose templates/seq_reads column from {specimen_label}"
                    )

                # Also extract Adaptive's V/J gene calls (-> "original_*" columns) and CDR3 AA call.
                # Follow tcrdist3's import_adaptive_file pattern (https://github.com/kmayerb/tcrdist3/blob/55d56fa621ec19b25a31ee1a3e61ef60e2575837/tcrdist/adpt_funcs.py#L24):
                # Don't just parse v_gene and j_gene; parse bio_identity instead.

                # Per https://tcrdist3.readthedocs.io/en/latest/adaptive.html:
                # "Adaptive’s output files can contain gene-level names within the ‘bioidentity’ field like TCRBV15-X, when there is ambiguity about the gene-level assignment."
                # Format example:
                # {'v_gene': 'unresolved',
                # 'v_gene_ties': 'TCRBV12-03/12-04,TCRBV12-04',
                # 'bio_identity': 'CATSAISSNQPQHF+TCRBV12-X+TCRBJ01-05'}
                df[["cdr3_seq_aa_q", "original_v_segment", "original_j_segment"]] = df[
                    "bio_identity"
                ].str.split("+", expand=True)

                # Trim CDR3: remove ends
                # and replace field that's entirely space (or empty) with NaN
                df.dropna(subset=["cdr3_seq_aa_q"], inplace=True)
                df["cdr3_seq_aa_q"] = (
                    df["cdr3_seq_aa_q"]
                    .str.slice(start=1, stop=-1)
                    .replace(r"^\s*$", np.nan, regex=True)
                )

            # Merge in igblast parses to get better V/J gene calls.
            parse_fnames = list(
                (fname.parent / "igblast_splits").glob(
                    f"split.{specimen_label}.*.parsed.tsv"
                )
            )
            if len(parse_fnames) == 0:
                raise ValueError(
                    f"No igblast parse files found for {specimen_label} from {study_name}"
                )
            df_parse = pd.concat(
                [pd.read_csv(fname, sep="\t") for fname in parse_fnames], axis=0
            )

            # extract fasta ID
            df_parse[["specimen_label", "rownum"]] = df_parse["id"].str.split(
                "|", expand=True
            )
            df_parse["rownum"] = df_parse["rownum"].astype(int)
            assert not df_parse["rownum"].duplicated().any()

            # For now we are assuming df and df_parse both have exactly one specimen
            assert (df_parse["specimen_label"] == specimen_label).all()

            if not is_adaptive:
                # get v_sequence (same way we produce v_sequence in internal pipeline's sort script)
                # this will be used to compute v_mut for BCR
                (
                    df_parse["v_sequence"],
                    df_parse["d_sequence"],
                    df_parse["j_sequence"],
                ) = get_v_sequence.complete_sequences(df_parse)
            else:
                # Adaptive->IgBlast reprocessing does not have the necessary sequence info for us,
                # because our Igblast-output parser fails to extract some sequence regions since Adaptive sequences are shorter.
                df_parse["v_sequence"] = pd.Series(dtype="str")
                df_parse["d_sequence"] = pd.Series(dtype="str")
                df_parse["j_sequence"] = pd.Series(dtype="str")

            orig_shape = df.shape
            df = pd.merge(
                df.rename(
                    # store original metadata as "original_*" columns
                    columns={
                        "productive": "original_productive",
                        "v_gene": "original_v_gene",
                        "j_gene": "original_j_gene",
                        "cdr3_seq_nt_q": "original_cdr3_seq_nt_q",
                        "cdr3_seq_aa_q": "original_cdr3_seq_aa_q",
                        # possible original Adaptive metadata:
                        "v_gene_ties": "original_v_gene_ties",
                        "j_gene_ties": "original_j_gene_ties",
                    }
                ),
                df_parse.set_index("rownum")[
                    [
                        "v_segment",
                        "j_segment",
                        "productive",
                        "v_sequence",
                        "d_sequence",
                        "j_sequence",
                    ]
                    + (
                        # Adaptive IgBlast rerun does not have sequences when interpreted by our parser,
                        # because Adaptive's sequences are shorter than ours
                        [
                            "cdr1_seq_aa_q",
                            "cdr2_seq_aa_q",
                            "cdr3_seq_nt_q",
                            "cdr3_seq_aa_q",
                        ]
                        if not is_adaptive
                        else []
                    )
                ],
                left_index=True,
                right_index=True,
                how="inner",
                validate="1:1",
            )
            assert df.shape[0] == min(orig_shape[0], df_parse.shape[0])

            # For Adaptive data, since CDR3 information was missing in IgBlast parses,
            # set cdr3_seq_nt_q and cdr3_seq_aa_q to the original values.
            if is_adaptive:
                df["cdr3_seq_nt_q"] = df["original_cdr3_seq_nt_q"]
                df["cdr3_seq_aa_q"] = df["original_cdr3_seq_aa_q"]
                # Also set other missing columns for consistency.
                # Not sure if these are true assumptions, but we will subset to TRBV V genes later, so should be ok
                df["locus"] = "TRB"
                df["extracted_isotype"] = "TCRB"

        # drop the external study's sequence_id (keep the int primary key instead)
        df = df.drop(columns="sequence_id", errors="ignore")

        # rename columns for consistency
        df.rename(
            columns={
                "c_call": "extracted_isotype",
                "id": "sequence_id",
            },
            inplace=True,
        )
        if gene_locus == GeneLocus.TCR:
            # sanity check
            if not (df["locus"] == "TRB").all():
                raise ValueError(
                    f"Locus field was not TRB for {specimen_label} ({study_name})"
                )
            # set isotype flag if iReceptor c_call is blank
            df["extracted_isotype"].fillna("TCRB", inplace=True)

        if study_name == "Montague":
            # extracted_isotype is not provided, but we know these are IgG
            if not all(df["extracted_isotype"].isna()):
                raise ValueError("We expect no isotype calls for Montague")
            df["extracted_isotype"].fillna("IGHG", inplace=True)

    # replace extracted_isotype values to have consistent prefix: IgG -> IGHG, IgA -> IGHA, etc.
    df["extracted_isotype"] = df["extracted_isotype"].str.replace(
        "^Ig", "IGH", regex=True
    )

    # filter
    df_orig_shape = df.shape
    df = df.loc[(~pd.isnull(df["extracted_isotype"])) & (df["productive"] == True)]
    if df.shape[0] == 0:
        raise ValueError(f"Filtering failed for {specimen_label} from {study_name}")

    # compute important columns
    # note that this converts v_segment, j_segment (with alleles) to v_gene, j_gene columns (no alleles).
    df = etl._compute_columns(df=df, gene_locus=gene_locus)

    # Replace indistinguishable TRBV gene names with the version that we use in our data.
    df["v_gene"] = df["v_gene"].replace(io.v_gene_renames)

    # since we are going to call clones by nucleotide sequences here rather than in the usual bioinformatics pipeline,
    # let's also preprocess the NT characters.
    # (note for Adaptive data: we had to remove the prefix/suffix from the CDR3 AA sequence.
    # we aren't bothering to do that with the nucleotide sequence, since we are just using that to set clone IDs
    # it's not being passed to the downstream language model, so it doesn't have to be consistent with the rest of our data.)
    df["cdr3_seq_nt_q_trim"] = etl._trim_sequences(df["cdr3_seq_nt_q"])
    df.dropna(
        subset=[
            "cdr3_seq_nt_q_trim",
        ],
        how="any",
        inplace=True,
    )

    # get trimmed lengths
    df["cdr3_nt_sequence_trim_len"] = df["cdr3_seq_nt_q_trim"].str.len()

    # Now that everything has gone through our IgBlast, these filters should be no-ops:
    # Downselect only to V genes that are in our standard dataset
    invalid_v_genes = set(df["v_gene"].unique()) - set(
        helpers.all_observed_v_genes()[gene_locus]
    )
    if len(invalid_v_genes) > 0:
        logger.warning(
            f"Dropping V genes from {specimen_label} ({study_name}) that aren't in our standard data: {invalid_v_genes}"
        )
        df = df.loc[df["v_gene"].isin(helpers.all_observed_v_genes()[gene_locus])]

    # And downselect only to J genes that are in our standard dataset
    invalid_j_genes = set(df["j_gene"].unique()) - set(
        helpers.all_observed_j_genes()[gene_locus]
    )
    if len(invalid_j_genes) > 0:
        logger.warning(
            f"Dropping J genes from {specimen_label} ({study_name}) that aren't in our standard data: {invalid_j_genes}"
        )
        df = df.loc[df["j_gene"].isin(helpers.all_observed_j_genes()[gene_locus])]

    # make each row a single unique VDJ sequence - drop duplicates
    # save number of reads to a column called `num_reads`, to measure clonal expansion later
    dedupe_cols = [
        "specimen_label",
        "extracted_isotype",
        "v_gene",
        "j_gene",
        "cdr1_seq_aa_q_trim",
        "cdr2_seq_aa_q_trim",
        "cdr3_seq_aa_q_trim",
        "cdr3_seq_nt_q_trim",
        "productive",
    ]
    if "num_reads" in df.columns:
        read_counts = (
            df.groupby(dedupe_cols, observed=True)["num_reads"].sum().reset_index()
        )
    else:
        read_counts = (
            df.groupby(dedupe_cols, observed=True).size().reset_index(name="num_reads")
        )
    df.drop_duplicates(
        subset=dedupe_cols,
        keep="first",
        inplace=True,
    )
    # sanity check
    if not all(df.groupby(dedupe_cols, observed=True).size() == 1):
        raise ValueError(f"Deduplicate failed for {specimen_label} from {study_name}")
    expected_shape = df.shape
    # merge in the num_reads column, replacing (dropping) the existing num_reads column if it exists
    df = pd.merge(
        df.drop(columns="num_reads", errors="ignore"),
        read_counts,
        how="left",
        on=dedupe_cols,
        validate="1:1",
    )
    if df.shape[0] != expected_shape[0]:
        raise ValueError(
            f"Merge post dedupe failed for {specimen_label} from {study_name}"
        )

    return df, (fname, df_orig_shape, df.shape, gene_locus)


# %%

# %%

# %%
all_specimens.head()

# %%

# %%

# %%
def run_on_all_specimens_from_a_person(participant_label, specimens):
    dfs = []
    info_per_specimen = []

    # This isn't true yet, but later on we might have BCR and TCR versions of the same specimen.
    # Get combined gene locus flag for all gene loci used in this group of specimens
    gene_loci = [GeneLocus[name] for name in specimens["gene_locus"].unique()]
    # combine flags
    gene_loci = GeneLocus.combine_flags_list_into_single_multiflag_value(gene_loci)
    GeneLocus.validate(gene_loci)

    # Load each specimen on its own. We expect one entry for BCR and another for TCR.
    for _, specimen in specimens.iterrows():
        gene_locus_for_this_specimen = GeneLocus[specimen["gene_locus"]]
        df, annotation = process_specimen(
            fname=specimen["fname"],
            gene_locus=gene_locus_for_this_specimen,
            study_name=specimen["study_name"],
            # note: passing in specimen label by locus. because on disk separated by (and named by) locus.
            specimen_label=specimen["specimen_label_by_locus"],
        )

        # assign metadata
        df = df.assign(
            participant_label=specimen["participant_label"],
            timepoint=specimen["timepoint"],
            is_peak=specimen["is_peak"],
            disease=specimen["disease"],
            disease_subtype=specimen["disease_subtype"],
            # at this point, rename specimen_label from current value (specimen_label_by_locus) to non-locus value (specimen_label)
            specimen_label=specimen["specimen_label"],
        )

        dfs.append((gene_locus_for_this_specimen, df))
        info_per_specimen.append(annotation)

    # cluster all specimens from this patient staple sequences from the specimens together.
    # but run clustering separately on different gene loci, because of different distance criteria.
    dfs_clustered_for_each_gene_locus = [
        ConvergentClusterClassifier._cluster_training_set(
            df=pd.concat([df for (gl, df) in dfs if gl == gene_locus], axis=0),
            sequence_identity_threshold=config.sequence_identity_thresholds.call_nucleotide_clones_with_patient[
                gene_locus
            ],
            validate_same_fold=False,
            higher_order_group_columns=[
                "v_gene",
                "j_gene",
                "cdr3_nt_sequence_trim_len",
            ],
            sequence_column="cdr3_seq_nt_q_trim",
            inplace=True,
        )
        for gene_locus in gene_loci
    ]

    # we can then combine clustering results across gene loci because V genes will be different (and V gene is included in cluster ID).
    df = pd.concat(dfs_clustered_for_each_gene_locus, axis=0)

    # Express global clone ID as string instead
    df.drop("global_resulting_cluster_ID", axis=1, inplace=True)
    df["igh_or_tcrb_clone_id"] = (
        df[
            [
                "specimen_label",
                "v_gene",
                "j_gene",
                "cdr3_nt_sequence_trim_len",
                "cluster_id_within_clustering_group",
            ]
        ]
        .astype(str)
        .apply(tuple, axis=1)
        .apply("|".join)
    )

    # Clustering also created a num_clone_members=1 column
    # Drop that so we can set it properly in sample_sequences
    df.drop("num_clone_members", axis=1, inplace=True)

    # Report total number of clones
    logger.info(
        f"Participant {participant_label} ({gene_loci}) has {df['igh_or_tcrb_clone_id'].nunique()} unique clones from specimens: {info_per_specimen}"
    )

    # Sample clones from each specimen,
    # with filters like dropping low-SHM naive B cells.
    # df may be empty after this
    df = pd.concat(
        [
            sample_sequences(specimen_df, required_gene_loci=gene_loci)
            for specimen_label, specimen_df in df.groupby("specimen_label")
        ],
        axis=0,
    )

    if df.shape[0] != 0:
        # Determine output filename: one file per external cohort participant.
        # Gene loci are combined into one file. They can be separated by the isotype column.
        fname_out = (
            config.paths.external_processed_data / f"{participant_label}.parquet"
        )
        if fname_out.exists():
            logger.warning(f"Path already exists, overwriting: {fname_out}")

        # Report number of sampled sequences
        logger.info(
            f"Participant {participant_label} ({gene_loci}) has {df.shape[0]} sampled sequences -> {fname_out}."
        )

        # Write
        df.to_parquet(fname_out, index=None)
        return fname_out
    else:
        logger.warning(
            f"Participant {participant_label} ({gene_loci}) had no sampled sequences. Skipping."
        )
        return None


# %%

# %%
# run on all specimens from same patient (combine all timepoints - even if mixed peak/non-peak status)
fnames_output = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
    delayed(run_on_all_specimens_from_a_person)(
        participant_label=participant_label, specimens=specimens
    )
    for participant_label, specimens in all_specimens.groupby(
        ["participant_label"], observed=True
    )
)

# %%

# %%
len(fnames_output)

# %%
# drop null returns
fnames_output = [f for f in fnames_output if f is not None]
len(fnames_output)

# %%

# %%
# Many, but not all, output fnames will exist now
len(fnames_output), all_specimens.shape[0]

# %%

# %%
pd.read_parquet(
    fnames_output[0],
).head()

# %%
pd.read_parquet(
    fnames_output[0],
).columns

# %%

# %%
