from typing import Union
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from malid import config, io, helpers, get_v_sequence
from malid.datamodels import GeneLocus

from tcrdist.adpt_funcs import _valid_cdr3
from malid.trained_model_wrappers import ConvergentClusterClassifier

logger = logging.getLogger(__name__)

# When we add new data formats, we should include a representative example in tests/test_etl.py.

## generate dtypes
# make dtype for exported "extracted_isotype" column more efficient using a categorical dtype
# see known_isotypes.ipynb
known_isotypes = [
    "IGHA",
    "IGHD",
    "IGHE",
    "IGHG1",
    "IGHG2",
    "IGHG3",
    "IGHG4",
    "IGHM",
    "IGHA1",
    "IGHA2",
    "multiIGHG2",
    "multiIGHG3",
    "gDNA",
]

# coalesce all subisotypes into isotype supergroups IgG, IgA, IgE, IgD/M and so on
# make sure these match helpers.py's isotype_groups_kept
map_isotype_to_isotype_supergroup = {
    "IGHA": "IGHA",
    "IGHA1": "IGHA",
    "IGHA2": "IGHA",
    "IGHD": "IGHD-M",
    "IGHE": "IGHE",
    "IGHG": "IGHG",
    "IGHG1": "IGHG",
    "IGHG2": "IGHG",
    "IGHG3": "IGHG",
    "IGHG4": "IGHG",
    "IGHM": "IGHD-M",
    "multiIGHG2": "IGHG",
    "multiIGHG3": "IGHG",
    "gDNA": "gDNA",
    "TCRB": "TCRB",
    # iReceptor c_call can include these values for TCR single cell data:
    "TRBC1": "TCRB",
    "TRBC2": "TCRB",
}

isotype_supergroups = np.unique(
    list(map_isotype_to_isotype_supergroup.values())
).tolist()

# make dtype for read in "isosubtype" column more efficient using a categorical dtype
# see known_isotypes.ipynb
known_isotypes_with_suffixes = pd.read_csv(
    config.paths.metadata_dir / "known_isotypes_with_suffixes.csv"
)["isosubtype"].unique()

# Subset of columns that we read in from participant-table CSVs on disk
common_dtypes_read_in = {
    # Shared columns between BCR and TCR:
    "participant_label": "category",
    "participant_age": "object",  # Int64
    "participant_description": "object",
    "specimen_label": "category",
    "specimen_time_point": "object",
    "amplification_locus": "category",
    "amplification_template": "category",
    # amplification_label further subdivides specimen_label, and replicate_label further subdivides amplification_label (see sample_sequences.py for more details).
    "amplification_label": "category",
    "replicate_label": "category",
    "run_label": "category",
    "trimmed_sequence": "object",
    "v_segment": "object",
    "j_segment": "object",
    "v_score": "float64",
    "productive": "category",  # productive's dtype will be adjusted below because we cast to bool
    # We ignore pre_seq_aa_q, which is only used for sequences that are so short that we did not observe some of the early framework or CDR regions.
    "fr1_seq_aa_q": "object",
    "cdr1_seq_aa_q": "object",
    "fr2_seq_aa_q": "object",
    "cdr2_seq_aa_q": "object",
    # FR1 through CDR2 will be unavailable for TCR data, based on our sequencing paradigm.
    # FR3, CDR3, and FR4 ("post") are available for both BCR and TCR.
    "fr3_seq_aa_q": "object",
    "cdr3_seq_aa_q": "object",
    "post_seq_aa_q": "object",
    "v_sequence": "object",
}
dtypes_read_in = {
    GeneLocus.BCR: {
        **common_dtypes_read_in,
        **{
            # Additional BCR-only columns
            "igh_clone_id": "int64",
            "spam_score": "float64",  # Not used. TODO: Remove.
            "isosubtype": pd.api.types.CategoricalDtype(known_isotypes_with_suffixes),
        },
    },
    GeneLocus.TCR: {
        **common_dtypes_read_in,
        **{
            # Additional TCR-only columns
            "tcrb_clone_id": "int64",
        },
    },
}

columns_dropped = ["igh_clone_id", "tcrb_clone_id"]

# add dtypes for columns we create
dtypes_newly_created = {
    "row_id_in_participant_table": "int64",
    "igh_or_tcrb_clone_id": "int64",
    "disease": "category",
    "disease_subtype": "category",
    "extracted_isotype": pd.api.types.CategoricalDtype(
        np.unique(known_isotypes + isotype_supergroups)
    ),
    "isotype_supergroup": pd.api.types.CategoricalDtype(isotype_supergroups),
    # Column with optional cell type identity values - added from metadata join
    "cell_type": "category",
    "num_reads": "int64",
    "v_gene": "category",
    "j_gene": "category",
    "fr1_seq_aa_q_trim": "object",
    "cdr1_seq_aa_q_trim": "object",
    "fr2_seq_aa_q_trim": "object",
    "cdr2_seq_aa_q_trim": "object",
    "fr3_seq_aa_q_trim": "object",
    "cdr3_seq_aa_q_trim": "object",
    "post_seq_aa_q_trim": "object",  # this is FR4
    "cdr3_aa_sequence_trim_len": "int64",
    "v_mut": "float64",
    # Here's a column we change from its dtype at read-in time: "productive" goes from being a general category to being a bool.
    # It starts as a category because Boydlab IgBlast parse exported from Postgres database has productive "t" and "f". Then in our code here, we cast to bool.
    # Not only should we mark the final dtype as bool for cleanliness, but Dask actually complains if we were to keep it as "category":
    # Dask will initialize "productive" as CategoricalDtype(categories=['__UNKNOWN_CATEGORIES__'], ordered=False)
    # Then when we provide bools for that column, we will get a warning from pyarrow/pandas_compat.py:611: FutureWarning: The dtype of the 'categories' of the passed categorical values (boolean) does not match the specified type (string). For now ignoring the specified type, but in the future this mismatch will raise a TypeError.
    "productive": "boolean",
}

# all of them put together
dtypes_expected_after_preprocessing = (
    dtypes_read_in[GeneLocus.BCR] | dtypes_read_in[GeneLocus.TCR] | dtypes_newly_created
)
dtypes_expected_after_preprocessing = {
    k: v
    for k, v in dtypes_expected_after_preprocessing.items()
    if k not in columns_dropped
}

# Load determinstic TCR sequence regions (generated by scripts/get_tcr_v_gene_annotations.py)
deterministic_sequence_regions = (
    pd.read_csv(config.paths.metadata_dir / "tcrb_v_gene_cdrs.generated.tsv", sep="\t")
    .rename(
        columns={
            "v_call": "v_segment",
            "fwr1_aa": "fr1_seq_aa_q",
            "cdr1_aa": "cdr1_seq_aa_q",
            "fwr2_aa": "fr2_seq_aa_q",
            "cdr2_aa": "cdr2_seq_aa_q",
            "fwr3_aa": "fr3_seq_aa_q",
        }
    )
    .set_index("v_segment")[
        [
            "fr1_seq_aa_q",
            "cdr1_seq_aa_q",
            "fr2_seq_aa_q",
            "cdr2_seq_aa_q",
            "fr3_seq_aa_q",
        ]
    ]
)
assert not deterministic_sequence_regions.index.duplicated().any()

allowed_hiv_runs = ["M111", "M112", "M113", "M114", "M124", "M125", "M132"]


def fix_dtypes(df, dtypes):
    """convert df dtypes (inplace) to match input dtypes, adding/removing columns as needed"""
    for col, dtype in dtypes.items():
        if col not in df.columns:
            # Create missing columns as empty.
            df[col] = pd.Series(dtype=dtype)
        else:
            # Set dtype of existing col.
            # Note that if dtype is a CategoricalDtype, and the column has entries that are not listed as categories of the dtype, those entries will become NaNs.
            df[col] = df[col].astype(dtype)

    # Drop columns that were not listed in the dtypes, i.e. subset to the intersection of dtypes and df.columns.
    # And return in order of dtypes.
    return df[list(dtypes.keys())]


def mutation_rate(series):
    """v_mut rate"""
    return (
        series.apply(lambda s: [c.isupper() for c in s]).apply(np.sum)
        / series.str.len()
    )


def read_boydlab_participant_table(
    fname: Union[str, Path], gene_locus: GeneLocus
) -> pd.DataFrame:
    dtypes = dtypes_read_in[gene_locus]
    cols = list(dtypes.keys())
    return pd.read_csv(fname, sep="\t", dtype=dtypes, usecols=cols)


def preprocess_each_participant_table(
    df: pd.DataFrame,
    gene_locus: GeneLocus,
    metadata_whitelist: pd.DataFrame,
):
    """
    Runs on a single person's "participant table" for a single sequencing locus (BCR or TCR).
    Involves: deduping and setting num_reads, setting extracted_isotype, setting disease and disease_subtype
    """
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)

    final_dtypes = dtypes_expected_after_preprocessing  # not dependent on locus

    def make_empty_output():
        """Helper function: return empty dataframe but with the right columns + dtypes"""
        return fix_dtypes(pd.DataFrame(), final_dtypes)

    # Store row ID in original participant table before we do anything else.
    # One-indexed, because 0th row coresponds to the original CSV header row.
    df["row_id_in_participant_table"] = range(1, df.shape[0] + 1)

    # Filter out anything except whitelisted specimen. This means df.shape[0] can become 0.
    # Important: metadata_whitelist's specimen_label must merge against the original (pre-override) specimen_label column loaded from the raw participant table.
    df = pd.merge(
        df,
        metadata_whitelist,
        how="inner",
        on=["participant_label", "specimen_label"],
    )
    # If we are dealing with an empty sample, stop here
    if df.shape[0] == 0:
        return make_empty_output()

    # Optionally override some label columns, using the "*_override" columns merged from metadata_whitelist
    # This can be used to consolidate specimens that were split into cell type fractions (e.g. CD4 vs CD8 T cells) with different specimen labels, which really are different fractions of the *same* specimen label.
    # (To be safe, allow these columns to not have been present in metadata_whitelist, though they should be)
    for col in [
        "participant_label",
        "specimen_label",
        "specimen_time_point",
        "amplification_label",
        "replicate_label",
    ]:
        if col in df and f"{col}_override" in df.columns:
            was_a_category = pd.api.types.is_categorical_dtype(df[col])
            df[col] = df[f"{col}_override"].fillna(df[col])
            if was_a_category:
                # make it a category again
                df[col] = df[col].astype("category")
            df.drop(columns=[f"{col}_override"], inplace=True)

    # Metadata_whitelist should also provide "hiv_run_filter" column in the merge (but check this as a condition in the boolean below, to be safe).
    # If this is a patient from the HIV cohort: allow specimens from certain runs only. ("hiv_run_filter" will be True for all rows from this person)
    # (We must check df.shape[0] > 0 so iloc[0] does not fail. Leaving this in as a boolean condition for clarity, though we already checked the shape above so it's not really necessary.)
    if (
        df.shape[0] > 0
        and "hiv_run_filter" in df.columns
        and df["hiv_run_filter"].iloc[0] == True
    ):
        # select certain run IDs only. exclude very old runs (M52 and such)
        # this means df.shape[0] can become 0
        df = df.loc[df["run_label"].isin(allowed_hiv_runs)]

    # Reset index
    df = df.reset_index(drop=True)

    # Consolidate columns
    if gene_locus == GeneLocus.BCR:
        df["igh_or_tcrb_clone_id"] = df["igh_clone_id"]
        expected_amplification_locus = "IgH"
    elif gene_locus == GeneLocus.TCR:
        df["igh_or_tcrb_clone_id"] = df["tcrb_clone_id"]
        expected_amplification_locus = "TCRB"
    else:
        raise ValueError(f"Unrecognized gene_locus: {gene_locus}")

    # Validate amplification_locus IgH or TCRB against provided gene_locus
    if not all(df["amplification_locus"] == expected_amplification_locus):
        raise ValueError(
            f"Expected amplification_locus {expected_amplification_locus} but found {df['amplification_locus'].unique()}"
        )

    ## Extract isotype
    if gene_locus == GeneLocus.BCR:
        df["extracted_isotype"] = (
            df["isosubtype"]
            .str.split("*")
            .str[0]
            .astype(
                pd.api.types.CategoricalDtype(
                    np.unique(known_isotypes + isotype_supergroups)
                )
            )
        )
        # set extracted_isotype to gDNA for gDNA runs
        df.loc[df["amplification_template"] == "gDNA", "extracted_isotype"] = "gDNA"
    elif gene_locus == GeneLocus.TCR:
        # set extracted_isotype to TCR for TCRB runs
        df["extracted_isotype"] = "TCRB"

    ## Filter
    # Igblast parse exported from Postgres database: cast to bool to be compatible
    df["productive"] = df["productive"].replace({"t": True, "f": False})
    df = df.loc[(~pd.isnull(df["extracted_isotype"])) & (df["productive"] == True)]
    if gene_locus == GeneLocus.BCR:
        # Additional BCR-only filter
        df = df.loc[df["v_score"] > 200]
    elif gene_locus == GeneLocus.TCR:
        # Additional TCR-only filter
        df = df.loc[df["v_score"] > 80]

    # If we are dealing with an empty sample, stop here
    if df.shape[0] == 0:
        return make_empty_output()

    ## dedupe to unique reads, to mostly correct for plasmablasts

    # save number of reads to a column called `num_reads`, to measure clonal expansion later
    # (use replicate_label instead of specimen_label to not be affected by consolidation of sample fractions like CD4/CD8)
    dedupe_cols = [
        "replicate_label",
        "trimmed_sequence",
        "extracted_isotype",
    ]

    read_counts = (
        df.groupby(dedupe_cols, observed=True).size().reset_index(name="num_reads")
    )

    df = df.drop_duplicates(dedupe_cols)
    # sanity check
    assert all(df.groupby(dedupe_cols, observed=True).size() == 1)

    # merge in the num_reads column
    df = pd.merge(df, read_counts, how="left", on=dedupe_cols, validate="1:1")

    # pre-compute some important columns
    df = _compute_columns(df=df, gene_locus=gene_locus)

    # Enforce specific dtype and column order again -- necessary if we have an empty sample by this point
    # (propogated the empty samples this far because we want all columns to be created, for concatenation into full dask dataframe)
    return fix_dtypes(df, final_dtypes)


def _split_post_seq_into_cdr3_and_fwr4(
    df: pd.DataFrame, cdr3_col: str, post_col: str
) -> pd.Series:
    """IgBlast run on Adaptive sequences incorrectly records CDR3 and (part of) FWR4 together as post_seq.
    Here we split the post column into CDR3 and FWR4.
    CDR3 is unchanged, so we just return the (partial) FWR4, which is the post column with the CDR3 prefixes removed.
    """
    # Combined sequence: post_col
    # Prefix sequence: cdr3_col
    # Edge cases: either one can be null; or post_col might not start with cdr3_col. If so, just return post_col.

    # First, let's trim the sequences so they are consistent
    # (This makes a copy)
    df = df.assign(
        **{
            cdr3_col: _trim_sequences(df[cdr3_col]),
            post_col: _trim_sequences(df[post_col]),
        }
    )

    # Subtract prefix_seq from combined_seq if it's at the start, and if both are not null
    processed_series = df.dropna(subset=[post_col, cdr3_col]).apply(
        lambda row: row[post_col][len(row[cdr3_col]) :]
        if row[post_col].startswith(row[cdr3_col])
        else row[post_col],
        axis=1,
    )

    # Now add back any ignored values from the original series (in case of edge cases)
    return processed_series.combine_first(df[post_col])


def _trim_cdr3s_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim CDR3: remove ends, and replace field that's entirely space (or empty) with NaN. Happens inplace"""
    # First, drop any entries with empty cdr3 aa: these seem to be nonproductive sequences.
    df.dropna(subset=["cdr3_seq_aa_q"], inplace=True)

    df["cdr3_seq_aa_q"] = (
        df["cdr3_seq_aa_q"]
        .str.slice(start=1, stop=-1)
        .replace(r"^\s*$", np.nan, regex=True)
    )

    # Drop any new NaNs (unlikely to have them)
    df.dropna(subset=["cdr3_seq_aa_q"], inplace=True)

    return df


def load_participant_data_external(
    participant_samples: pd.DataFrame,
    gene_locus: GeneLocus,
    base_path: Path,
    is_adaptive: bool,
    expect_a_read_count_column: bool = True,
    file_extension: str = "tsv",
) -> pd.DataFrame:
    """
    Input: a single participant's samples (dataframe where each row is a sample), for a single sequencing locus (BCR or TCR).

    If Adaptive data: We expect ImmuneAcccess Export Sample v2 format.

    Output: standardized format, including deduping and setting num_reads, setting extracted_isotype, setting disease and disease_subtype
    """

    # Adaptive, reprocessed through our IgBlast
    # See https://www.adaptivebiotech.com/wp-content/uploads/2019/07/MRK-00342_immunoSEQ_TechNote_DataExport_WEB_REV.pdf

    # Our IgBlast gives some different V gene calls, but generally doesn't provide CDR3 calls for these short sequences.
    # That's because our parser looks for the location of our primers.
    # We'll use the V/J gene and productive calls from our IgBlast, while using Adaptive's CDR3 call.

    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)

    if is_adaptive and gene_locus != GeneLocus.TCR:
        raise NotImplementedError("Only TCR implemented for Adaptive data")

    assert participant_samples["participant_label"].nunique() == 1
    assert participant_samples["study_name"].nunique() == 1
    participant_label, study = (
        participant_samples["participant_label"].iloc[0],
        participant_samples["study_name"].iloc[0],
    )

    # Special case a study that has no raw sequences available to run IgBlast ourselves.
    # Our "fill in determinstic TCR regions" code doesn't handle FR4 (post_seq_aa_q) yet, so we will disable checks of that field for this study.
    allow_blank_fr4_region: bool = study == "Britanova" and gene_locus == GeneLocus.TCR

    # Load all samples
    dfs = []
    for _, row in participant_samples.iterrows():
        # Load single sample.
        sample_name = row["sample_name"]
        fname_sample = base_path / study / f"{sample_name}.{file_extension}"
        df = pd.read_csv(
            fname_sample,
            sep="," if "csv" in file_extension else "\t",
            # special N/A values for Adaptive data
            na_values=["no data", "unknown", "unresolved", "na", "null"],
            # Solve "DtypeWarning: Columns ... have mixed types. Specify dtype option on import or set low_memory=False."
            dtype={
                "dGeneName": "object",
                "dGeneNameTies": "object",
                "jFamilyTies": "object",
                "jGeneNameTies": "object",
                "jGeneAlleleTies": "object",
                "vFamilyTies": "object",
            },
        ).assign(sample_name=sample_name)

        if df.shape[0] == 0:
            # Skip empty sample
            logger.warning(f"Skipping empty sample {sample_name} in study {study}")
            continue

        if study == "Britanova" and gene_locus == GeneLocus.TCR:
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

        # Not done here, but we could bring back:
        # Define whether productive from Adaptive metadata by checking whether "frame_type" or "sequenceStatus" column equals "In"

        # Count copies using whichever field is available.
        # Possible column names:
        # "count (templates/reads)": immuneaccess v2
        # "templates": immunecode
        # "seq_reads": Emerson
        for col in ["count (templates/reads)", "templates", "reads", "seq_reads"]:
            if col in df.columns and not df[col].isna().any():
                # We have found a read count column. Set it to the standard name.
                df["num_reads"] = df[col]
                break
        if "num_reads" not in df.columns and expect_a_read_count_column:
            # Note that some studies are exempt from having a read count column.
            raise ValueError(
                f"No template/read count column found for sample {sample_name} in study {study}"
            )
        if "num_reads" in df.columns and not expect_a_read_count_column:
            # If we are handling a study that is exempt from having a read count column, then we should not have found one.
            raise ValueError(
                f"Unexpected template/read count column found for sample {sample_name} in study {study}"
            )

        # Also extract Adaptive's CDR3 nucleotide and amino acid sequences.
        if is_adaptive:
            # Get nucleotide sequence, which may have several different column names. Try them until we find one.
            for possible_column_name in [
                "cdr3_rearrangement",  # emerson
                # The column names below are for the entire detected rearrangement, not just cdr3, but will suffice for finding nucleotide uniques.
                # Ideally we'd use CDR3 nt sequence only, but that's not available in formats like ImmuneAccess v2.
                "rearrangement",  # immunecode
                "nucleotide",  # immuneaccess
            ]:
                if possible_column_name in df.columns:
                    df.rename(
                        columns={possible_column_name: "cdr3_seq_nt_q"}, inplace=True
                    )
                    break
            if "cdr3_seq_nt_q" not in df.columns:
                raise ValueError(
                    f"No nucleotide rearrangement column found for sample {sample_name} in study {study}"
                )

            # Repeat for amino acid sequence
            for possible_column_name in [
                "amino_acid",  # emerson, immunecode
                "aminoAcid",  # immuneaccess
            ]:
                if possible_column_name in df.columns:
                    df.rename(
                        columns={possible_column_name: "cdr3_seq_aa_q"}, inplace=True
                    )
                    break
            if "cdr3_seq_aa_q" not in df.columns:
                raise ValueError(
                    f"No CDR3 amino acid sequence column found for sample {sample_name} in study {study}"
                )

        if study == "Britanova" and gene_locus == GeneLocus.TCR:
            # Special case for this study:
            # No raw sequences available to run IgBlast ourselves.

            # Pulling in CDR1 and CDR2 for TCRB data in etl._compute_columns expects v_segment to be set with allele info.
            # etl._compute_columns guarantees that each V gene has a CDR1 and CDR2 available for its dominant *01 allele.
            df["v_segment"] = df["v_gene"] + "*01"
            df["j_segment"] = df["j_gene"] + "*01"

            # Trim CDR3: remove ends, and replace field that's entirely space (or empty) with NaN.
            df = _trim_cdr3s_inplace(df)

            # Mark these all as productive TCRB
            df["productive"] = True
            df["extracted_isotype"] = "TCRB"

            # No IgBlast parse merging to do. Proceed to next sample.
            dfs.append(df)
            continue

        ### Merge IgBlast parses:

        # Construct IgBlast parse filenames (split into files of 10k sequences)
        parse_fnames = list(
            (base_path / study).glob(
                f"{sample_name}.fasta.*.parsed.{'TCRB' if gene_locus == GeneLocus.TCR else 'IgH'}.tsv"
            )
        )
        if len(parse_fnames) == 0:
            raise ValueError(
                f"No igblast parse files found for study {study}, sample {sample_name}"
            )

        # Merge in IgBlast parses for better V/J gene calls and other fields.
        # First, load the IgBlast parses (split into files of 10k sequences)
        df_parse = pd.concat(
            [
                pd.read_csv(
                    fname,
                    sep="\t",
                    # Solve "DtypeWarning: Columns ... have mixed types. Specify dtype option on import or set low_memory=False."
                    dtype={
                        "pre_seq_nt_q": "object",
                        "pre_seq_nt_v": "object",
                        "pre_seq_nt_d": "object",
                        "pre_seq_nt_j": "object",
                        "fr1_seq_nt_q": "object",
                        "fr1_seq_nt_v": "object",
                        "fr1_seq_nt_d": "object",
                        "fr1_seq_nt_j": "object",
                        "cdr1_seq_nt_q": "object",
                        "cdr1_seq_nt_v": "object",
                        "cdr1_seq_nt_d": "object",
                        "cdr1_seq_nt_j": "object",
                        "fr2_seq_nt_q": "object",
                        "fr2_seq_nt_v": "object",
                        "fr2_seq_nt_d": "object",
                        "fr2_seq_nt_j": "object",
                        "cdr2_seq_nt_q": "object",
                        "cdr2_seq_nt_v": "object",
                        "cdr2_seq_nt_d": "object",
                        "cdr2_seq_nt_j": "object",
                        "fr3_seq_nt_q": "object",
                        "fr3_seq_nt_v": "object",
                        "fr3_seq_nt_d": "object",
                        "fr3_seq_nt_j": "object",
                        "cdr3_seq_nt_q": "object",
                        "cdr3_seq_nt_v": "object",
                        "cdr3_seq_nt_d": "object",
                        "cdr3_seq_nt_j": "object",
                        "post_seq_nt_q": "object",
                        "post_seq_nt_v": "object",
                        "post_seq_nt_d": "object",
                        "post_seq_nt_j": "object",
                        "pre_seq_aa_q": "object",
                        "fr1_seq_aa_q": "object",
                        "cdr1_seq_aa_q": "object",
                        "fr2_seq_aa_q": "object",
                        "cdr2_seq_aa_q": "object",
                        "fr3_seq_aa_q": "object",
                        "cdr3_seq_aa_q": "object",
                        "post_seq_aa_q": "object",
                        # pandas boolean dtype is nullable:
                        "productive": "boolean",
                        "v_j_in_frame": "boolean",
                    },
                )
                for fname in parse_fnames
            ],
            axis=0,
        )

        # extract fasta ID
        df_parse[["sample_name", "rownum"]] = df_parse["id"].str.split("|", expand=True)
        df_parse["rownum"] = df_parse["rownum"].astype(int)
        assert not df_parse["rownum"].duplicated().any()

        # For now we are assuming df and df_parse both have exactly one sample
        assert (df_parse["sample_name"] == sample_name).all()

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
            # Drop possible original columns so we don't confuse with the IgBlast results
            df.drop(
                columns=[
                    "repertoire_id",
                    "sequence_id",  # TODO: What is this
                    "productive",
                    "cdr1_amino_acid",
                    "cdr1_rearrangement_length",
                    "cdr1_rearrangement",
                    "cdr1_start_index",
                    "cdr2_amino_acid",
                    "cdr2_rearrangement_length",
                    "cdr2_rearrangement",
                    "cdr2_start_index",
                    "cdr3_amino_acid",
                    "cdr3_length",
                    "cdr3_rearrangement_length",
                    "cdr3_rearrangement",
                    "cdr3_start_index",
                    "chosen_j_allele",
                    "chosen_j_family",
                    "chosen_j_gene",
                    "chosen_v_allele",
                    "chosen_v_family",
                    "chosen_v_gene",
                    "d_allele_ties",
                    "d_allele",
                    "d_family_ties",
                    "d_family",
                    "d_gene_ties",
                    "d_gene",
                    "d_index",
                    "d_resolved",
                    "d3_deletions",
                    "d5_deletions",
                    "dj_rearrangements",
                    "dj_templates",
                    "extended_rearrangement",
                    "frame_type",
                    "j_allele_ties",
                    "j_allele",
                    "j_deletions",
                    "j_family_ties",
                    "j_family",
                    "j_gene_ties",
                    "j_gene",
                    "j_index",
                    "j_resolved",
                    "n1_index",
                    "n1_insertions",
                    "n2_index",
                    "n2_insertions",
                    "outofframe_rearrangements",
                    "outofframe_templates",
                    "rearrangement_trunc",
                    "rearrangement",
                    "stop_rearrangements",
                    "v_allele_ties",
                    "v_allele",
                    "v_deletions",
                    "v_family_ties",
                    "v_family",
                    "v_gene_ties",
                    "v_gene",
                    "v_index",
                    "v_resolved",
                    "v_shm_count",
                    "v_shm_indexes",
                ]
                + (["cdr3_seq_nt_q", "cdr3_seq_aa_q"] if not is_adaptive else []),
                errors="ignore",
            ),
            df_parse.set_index("rownum")[
                [
                    "v_segment",
                    "j_segment",
                    "productive",
                    # These will be blank if Adaptive:
                    "v_sequence",
                    "d_sequence",
                    "j_sequence",
                    # These will not exist in Adaptive:
                    # "fr1_seq_aa_q",
                    # "cdr1_seq_aa_q",
                    # "fr2_seq_aa_q",
                    # "cdr2_seq_aa_q",
                    # The reason why the above are missing is that Adaptive->IgBlast reprocessing does not have the necessary sequence info for us,
                    # because our Igblast-output parser fails to extract some sequence regions since Adaptive sequences are shorter than ours.
                    # So we can't extract FR1->CDR2 from igblast if Adaptive.
                    # What can we extract from IgBlast?
                    # 1) the V and J gene calls
                    # 2) the FWR3 and FWR4 from those Adaptive sequences.
                    # However, the CDR3 has been swallowed into the FWR4 "post_seq" field.
                    # We will post-process this shortly to get the true FWR4.
                    "fr3_seq_aa_q",
                    "post_seq_aa_q",
                    # For Adaptive data, since CDR3 information was missing in IgBlast parses,
                    # we keep the original cdr3_seq_aa_q value direct from Adaptive.
                ]
                + (
                    [
                        # These will be missing in Adaptive:
                        "fr1_seq_aa_q",
                        "cdr1_seq_aa_q",
                        "fr2_seq_aa_q",
                        "cdr2_seq_aa_q",
                        #
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
        expected_len = min(orig_shape[0], df_parse.shape[0])
        if df.shape[0] != expected_len:
            # Sometimes the IgBlast parser rejects malformed sequences: they might be in df but not in df_parse.
            # Allow a tolerance for df to drop in size by at most 2%. Otherwise throw an error.
            if not (
                df.shape[0] < expected_len
                and np.isclose(df.shape[0], expected_len, rtol=0.02)
            ):
                raise ValueError(
                    f"IgBlast merge failed for study {study}, sample {sample_name}: {df.shape[0]} != min({orig_shape[0]}, {df_parse.shape[0]})"
                )

        # Trim CDR3: remove ends, and replace field that's entirely space (or empty) with NaN.
        df = _trim_cdr3s_inplace(df)

        if is_adaptive:
            # Post-process the FWR4 to remove the CDR3 prefix:
            # IgBlast mis-annotated the CDR3+FWR4 as "post_seq" together.
            # We need to split them apart.
            df["post_seq_aa_q"] = _split_post_seq_into_cdr3_and_fwr4(
                df, "cdr3_seq_aa_q", "post_seq_aa_q"
            )

        # rename columns for consistency
        df.rename(
            columns={
                "c_call": "extracted_isotype",
                "isotype": "extracted_isotype",  # Briney format
                "id": "sequence_id",
            },
            inplace=True,
        )

        # Fill isotype flag in situations where we don't have it but know the value.
        # e.g. TCRB data from Adaptive, where we don't have an isotype call.
        if gene_locus == GeneLocus.TCR:
            if "extracted_isotype" in df.columns:
                df["extracted_isotype"].fillna("TCRB", inplace=True)
            else:
                df["extracted_isotype"] = "TCRB"
        # this is also the place to special case on the variable `study`, for any BCR studies that don't have isotype calls but we know they are all IGHG for example.
        if study == "Montague":
            # extracted_isotype is not provided, but we know these are IgG
            if not all(df["extracted_isotype"].isna()):
                raise ValueError("We expect no isotype calls for Montague")
            df["extracted_isotype"].fillna("IGHG", inplace=True)

        dfs.append(df)

    # We have now loaded each sample from this person.

    # Combine all samples from this person
    if len(dfs) == 0:
        logger.warning(
            f"All samples for study {study}, participant {participant_label} were empty. Skipping."
        )
        # Return empty dataframe but with the right columns + dtypes
        return fix_dtypes(pd.DataFrame(), dtypes_expected_after_preprocessing)

    df = pd.concat(dfs, axis=0)
    del dfs

    # Confirm isotype column found (BCR) or created (TCR)
    if "extracted_isotype" not in df.columns:
        raise ValueError("extracted_isotype column not found")

    # Merge in metadata for each sample_name
    # This will introduce replicate_label and specimen_label columns, allowing us to consolidate specimens that were split into cell type fractions (e.g. CD4 vs CD8 T cells)
    # (It will also introduce amplification_label, which is usually the same as specimen_label for Adaptive data - see the notes in sample_sequences.py for the nuance of this.)
    # Later, we can extend this to merging on "sample_name" and "locus" if we have BCR+TCR external cohorts.
    df = pd.merge(
        df, participant_samples, how="left", on=["sample_name"], validate="m:1"
    )

    # We don't have this concept here
    df["row_id_in_participant_table"] = -1

    # Filter
    df = df.loc[(~pd.isnull(df["extracted_isotype"])) & (df["productive"] == True)]
    if df.shape[0] == 0:
        raise ValueError(
            f"Filtering failed for study {study}, participant {participant_label}"
        )

    # Replace extracted_isotype values to have consistent prefix: IgG -> IGHG, IgA -> IGHA, etc.
    df["extracted_isotype"] = df["extracted_isotype"].str.replace(
        "^Ig", "IGH", regex=True
    )
    if not all(df["extracted_isotype"].isin(map_isotype_to_isotype_supergroup.keys())):
        raise ValueError(
            f"Unexpected isotypes found for study {study}, participant {participant_label}: {df['extracted_isotype'].unique()}"
        )

    # Compute important columns
    # Note that this converts v_segment, j_segment (with alleles) to v_gene, j_gene columns (no alleles).
    # This also fills in sequence regions that we did not directly sequence, using the germline reference.
    # It also overwrites FR3, which we only sequenced partially.
    df = _compute_columns(
        df=df, gene_locus=gene_locus, skip_fr4_region_validation=allow_blank_fr4_region
    )

    # since we are going to call clones by nucleotide sequences here rather than in the usual bioinformatics pipeline,
    # let's also preprocess the NT characters.
    # the amino acid columns already got trimmed in _compute_columns.
    # (note for Adaptive data: we had to remove the prefix/suffix from the CDR3 AA sequence.
    # we aren't bothering to do that with the nucleotide sequence, since we are just using that to set clone IDs
    # it's not being passed to the downstream language model, so it doesn't have to be consistent with the rest of our data.)
    df["cdr3_seq_nt_q_trim"] = _trim_sequences(df["cdr3_seq_nt_q"])
    df.dropna(
        subset=[
            "cdr3_seq_nt_q_trim",
        ],
        how="any",
        inplace=True,
    )

    # get trimmed lengths
    df["cdr3_nt_sequence_trim_len"] = df["cdr3_seq_nt_q_trim"].str.len()

    # Extra nomenclature filter for studies that were not processed through our IgBlast.
    # For every other study that has gone through our IgBlast, these filters should be no-ops.
    # Downselect only to V genes that are in our standard dataset:
    invalid_v_genes = set(df["v_gene"].unique()) - set(
        helpers.all_observed_v_genes()[gene_locus]
    )
    if len(invalid_v_genes) > 0:
        logger.warning(
            f"Dropping V genes from participant {participant_label} (study {study}) that aren't in our standard data: {invalid_v_genes}"
        )
        df = df.loc[df["v_gene"].isin(helpers.all_observed_v_genes()[gene_locus])]

    # And downselect only to J genes that are in our standard dataset:
    invalid_j_genes = set(df["j_gene"].unique()) - set(
        helpers.all_observed_j_genes()[gene_locus]
    )
    if len(invalid_j_genes) > 0:
        logger.warning(
            f"Dropping J genes from participant {participant_label} (study {study}) that aren't in our standard data: {invalid_j_genes}"
        )
        df = df.loc[df["j_gene"].isin(helpers.all_observed_j_genes()[gene_locus])]

    ## dedupe to unique reads, to mostly correct for plasmablasts
    # make each row a single unique VDJ sequence - drop duplicates
    # save number of reads to a column called `num_reads`, to measure clonal expansion later
    # (use replicate_label instead of specimen_label to not be affected by consolidation of sample fractions like CD4/CD8)
    dedupe_cols = [
        "replicate_label",
        "extracted_isotype",
        # the below columns define the sequence:
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
        raise ValueError(
            f"Deduplicate failed for study {study}, participant {participant_label}"
        )
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
            f"Merge post dedupe failed for study {study}, participant {participant_label}"
        )

    # Assign clone IDs:
    # Cluster sequences from this participant (across all specimens).
    # (but run clustering separately on different gene loci, because of different distance criteria.
    # we can then combine clustering results across gene loci because V genes will be different and V gene is included in cluster ID.
    # this is automatic for us since TCR only for now.)
    df = ConvergentClusterClassifier._cluster_training_set(
        df=df,
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

    # Express global clone ID (igh_or_tcrb_clone_id) as int64 instead, the expected dtype for our ETL flow.
    # These are clone IDs within each participant, across all specimens (i.e. all timepoints, mixing peak and non-peak samples -- and including all replicates of those timepoints) from that person.
    # Clone IDs are not unique across patients.
    df["igh_or_tcrb_clone_id"] = (
        df["global_resulting_cluster_ID"].astype("category").cat.codes
    )
    df.drop("global_resulting_cluster_ID", axis=1, inplace=True)

    # Clustering also created a num_clone_members=1 column
    # Drop that so we can set it properly in sample_sequences
    df.drop("num_clone_members", axis=1, inplace=True)

    # Enforce specific dtype and column order again -- necessary if we have an empty sample by this point
    # (propogated the empty samples this far because we want all columns to be created, for concatenation into full dask dataframe)
    return fix_dtypes(df, dtypes_expected_after_preprocessing)


def _compute_columns(
    df: pd.DataFrame, gene_locus: GeneLocus, skip_fr4_region_validation: bool = False
) -> pd.DataFrame:
    df = df.copy()

    # Coalesce subisotypes into isotype "supergroup"
    df["isotype_supergroup"] = (
        df["extracted_isotype"]
        .map(map_isotype_to_isotype_supergroup)
        .astype(pd.api.types.CategoricalDtype(isotype_supergroups))
    )
    if df["isotype_supergroup"].isna().any():
        raise ValueError(
            f"Unexpected NaNs in isotype_supergroup, likely due to unexpected isotype values: {df['extracted_isotype'].unique()}"
        )

    # rename wrongly-named alleles in special cases
    df["v_segment"] = df["v_segment"].replace(io.v_allele_renames)

    # Pull out v, j genes (drop alleles)
    # (for V gene: Defer category conversion until after we have renamed indistinguishable TRBV gene names below)
    df["v_gene"] = df["v_segment"].str.split("*").str[0]
    df["j_gene"] = df["j_segment"].str.split("*").str[0].astype("category")

    # Replace indistinguishable TRBV gene names with the version that we use in our data.
    df["v_gene"] = df["v_gene"].replace(io.v_gene_renames).astype("category")

    # Regions prior to FR3 are not included in TCR data because of the way we do sequencing.
    # We still want to embed them to include V gene context in language model.
    # TCRs has deterministic CDR1 and CDR2 because no SHM. So let's fill them.
    if gene_locus == GeneLocus.TCR:
        # Drop empty columns (if they exist)
        # Also drop fr3_seq_aa_q, which is not empty, but which we likely have only sequenced part of.
        # We will deterministically fill in FR1->CDR2 and overwrite FR3.
        # TODO: Can we also deterministically fill in FR4? Once we do that, remove skip_fr4_region_validation flag.
        df.drop(
            columns=[
                "fr1_seq_aa_q",
                "cdr1_seq_aa_q",
                "fr2_seq_aa_q",
                "cdr2_seq_aa_q",
                "fr3_seq_aa_q",
            ],
            errors="ignore",
            inplace=True,
        )

        # Join on V gene + allele (note: not just V gene!)
        df = pd.merge(
            df,
            deterministic_sequence_regions,
            how="left",
            left_on="v_segment",
            right_index=True,
        )

        # Log which ones are lost, and how many rows
        if df["cdr1_seq_aa_q"].isna().all():
            raise ValueError(
                "Failed to merge TCRB CDR1/CDR2 by v_segment. Make sure we provide v_segment with allele included. It's ok to add *01 allele because we are guaranteed sequence availability for the dominant allele."
            )
        missing_entries = (
            df["cdr1_seq_aa_q"]
            .isna()
            .groupby(df["v_segment"], observed=True)
            .sum()
            .rename("number of rows missing a CDR1")
        )
        missing_entries = missing_entries[missing_entries > 0]
        if len(missing_entries) > 0:
            logger.info(
                f"Number of rows missing a CDR1, by v_segment: {missing_entries.to_dict()}"
            )

    # Trim AA strings
    # Optionally exclude post region for special studies that don't have it, since we are not yet deterministically filling this TCR region.
    for aa_col in [
        "fr1_seq_aa_q",
        "cdr1_seq_aa_q",
        "fr2_seq_aa_q",
        "cdr2_seq_aa_q",
        "fr3_seq_aa_q",
        "cdr3_seq_aa_q",
    ] + ([] if skip_fr4_region_validation else ["post_seq_aa_q"]):
        trimmed_aa_col = f"{aa_col}_trim"
        if aa_col in df.columns:
            col_to_use = aa_col
        elif trimmed_aa_col in df.columns:
            # handle edge case where we don't have _aa_q but already have _aa_q_trim
            col_to_use = trimmed_aa_col
        else:
            raise ValueError(
                f"Neither {aa_col} nor {trimmed_aa_col} are in the dataframe. Cannot trim."
            )
        df[trimmed_aa_col] = _trim_sequences(df[col_to_use])

    # Drop rows with any empty CDRs
    # Optionally exclude post region for special studies that don't have it, since we are not yet deterministically filling this TCR region.
    required_amino_acid_cols = [
        "fr1_seq_aa_q_trim",
        "cdr1_seq_aa_q_trim",
        "fr2_seq_aa_q_trim",
        "cdr2_seq_aa_q_trim",
        "fr3_seq_aa_q_trim",
        "cdr3_seq_aa_q_trim",
    ] + ([] if skip_fr4_region_validation else ["post_seq_aa_q_trim"])
    df.dropna(
        subset=["v_gene", "j_gene"] + required_amino_acid_cols,
        how="any",
        inplace=True,
    )

    # Ensure all amino acids are from the amino acid alphabet
    for amino_acid_column in required_amino_acid_cols:
        df = df.loc[df[amino_acid_column].apply(lambda seq: _valid_cdr3(seq))]

    # Calculate derived properties
    df["cdr3_aa_sequence_trim_len"] = df["cdr3_seq_aa_q_trim"].str.len()
    # v_mut should only be important for BCR
    df["v_mut"] = (
        mutation_rate(df["v_sequence"]) if gene_locus == GeneLocus.BCR else 0.0
    )

    return df


def _trim_sequences(sequences: pd.Series) -> pd.Series:
    trimmed_seqs = (
        sequences.str.replace(".", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.strip()
        .str.upper()
    )
    # change empty strings to NaN
    return trimmed_seqs.mask(trimmed_seqs == "")
