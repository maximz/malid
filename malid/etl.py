import numpy as np
import pandas as pd
import logging

from malid import config, helpers, io
from malid.datamodels import GeneLocus

from tcrdist.adpt_funcs import _valid_cdr3

logger = logging.getLogger(__name__)

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

# another categorical dtype
known_diseases = helpers.diseases

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
    "run_label": "category",
    "trimmed_sequence": "object",
    "v_segment": "object",
    "j_segment": "object",
    "v_score": "float64",
    "productive": "category",
    "cdr1_seq_aa_q": "object",
    "cdr2_seq_aa_q": "object",
    "cdr3_seq_aa_q": "object",
    "v_sequence": "object",
}
dtypes_read_in = {
    GeneLocus.BCR: {
        **common_dtypes_read_in,
        **{
            # Additional BCR-only columns
            "igh_clone_id": "int64",
            "spam_score": "float64",
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
    "igh_or_tcrb_clone_id": "int64",
    "disease": pd.api.types.CategoricalDtype(known_diseases),
    "disease_subtype": "category",
    "extracted_isotype": pd.api.types.CategoricalDtype(
        np.unique(known_isotypes + isotype_supergroups)
    ),
    "isotype_supergroup": pd.api.types.CategoricalDtype(isotype_supergroups),
    "num_reads": "int64",
    "v_gene": "category",
    "j_gene": "category",
    "cdr1_seq_aa_q_trim": "object",
    "cdr2_seq_aa_q_trim": "object",
    "cdr3_seq_aa_q_trim": "object",
    "cdr3_aa_sequence_trim_len": "int64",
    "v_mut": "float64",
}

# all of them put together
dtypes_expected_after_preprocessing = {
    **dtypes_read_in[GeneLocus.BCR],
    **dtypes_read_in[GeneLocus.TCR],
    **dtypes_newly_created,
}
dtypes_expected_after_preprocessing = {
    k: v
    for k, v in dtypes_expected_after_preprocessing.items()
    if k not in columns_dropped
}


def fix_dtypes(df, dtypes):
    """convert df dtypes (inplace) to match input dtypes, adding/removing columns as needed"""
    for col, dtype in dtypes.items():
        if col not in df.columns:
            # Create missing columns as empty.
            df[col] = pd.Series(dtype=dtype)
        else:
            # Set dtype of existing col.
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


def preprocess_each_participant_table(
    df: pd.DataFrame, gene_locus: GeneLocus, final_dtypes: dict
):
    """deduping and setting num_reads, setting extracted_isotype, setting disease and disease_subtype"""
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)

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
    df = df.loc[(~pd.isnull(df["extracted_isotype"])) & (df["productive"] == "t")]
    if gene_locus == GeneLocus.BCR:
        # Additional BCR-only filter
        df = df.loc[(df["v_score"] > 200) & (df["spam_score"] <= 0.0)]
    elif gene_locus == GeneLocus.TCR:
        # Additional TCR-only filter
        df = df.loc[df["v_score"] > 80]

    # If we are dealing with an empty sample, stop here
    if df.shape[0] == 0:
        # return empty dataframe but with the right columns + dtypes
        return fix_dtypes(pd.DataFrame(), final_dtypes)

    ## dedupe to unique reads, to mostly correct for plasmablasts

    # save number of reads to a column called `num_reads`, to measure clonal expansion later
    dedupe_cols = ["specimen_label", "trimmed_sequence", "extracted_isotype"]

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


def _compute_columns(df: pd.DataFrame, gene_locus: GeneLocus):
    # Coalesce subisotypes into isotype "supergroup"
    df["isotype_supergroup"] = (
        df["extracted_isotype"]
        .replace(map_isotype_to_isotype_supergroup)
        .astype(pd.api.types.CategoricalDtype(isotype_supergroups))
    )

    # rename wrongly-named alleles in special cases
    df["v_segment"] = df["v_segment"].replace(io.v_allele_renames)

    # pull out v, j genes (drop alleles)
    df["v_gene"] = df["v_segment"].str.split("*").str[0].astype("category")
    df["j_gene"] = df["j_segment"].str.split("*").str[0].astype("category")

    # CDR1 and CDR2 are not included in TCR data (why?)
    # We still want to embed them to include V gene context in language model.
    # TCRs has deterministic CDR1 and CDR2 because no SHM. So let's fill them.
    if gene_locus == GeneLocus.TCR:
        # Drop empty columns (if they exist)
        df.drop(
            columns=["cdr1_seq_aa_q", "cdr2_seq_aa_q"], errors="ignore", inplace=True
        )

        # Load (generated by scripts/get_tcr_v_gene_annotations.py)
        determinstic_cdrs = (
            pd.read_csv(
                config.paths.metadata_dir / "tcrb_v_gene_cdrs.generated.tsv", sep="\t"
            )
            .rename(
                columns={
                    "v_call": "v_segment",
                    "cdr1_aa": "cdr1_seq_aa_q",
                    "cdr2_aa": "cdr2_seq_aa_q",
                }
            )
            .drop(columns=["v_gene", "v_allele"])
            .set_index("v_segment")
        )
        assert not determinstic_cdrs.index.duplicated().any()

        # Join on V gene + allele (note: not just V gene!)
        df = pd.merge(
            df, determinstic_cdrs, how="left", left_on="v_segment", right_index=True
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

    # trim AA and NT characters (for some NTs)
    for aa_col in [
        "cdr1_seq_aa_q",
        "cdr2_seq_aa_q",
        "cdr3_seq_aa_q",
    ]:
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
    required_cols = [
        "v_gene",
        "j_gene",
        "cdr1_seq_aa_q_trim",
        "cdr2_seq_aa_q_trim",
        "cdr3_seq_aa_q_trim",
    ]
    df.dropna(
        subset=required_cols,
        how="any",
        inplace=True,
    )

    # Confirm all CDR1, CDR2, and CDR3 amino acids are from the amino acid alphabet
    df = df.loc[df["cdr1_seq_aa_q_trim"].apply(lambda cdr: _valid_cdr3(cdr))]
    df = df.loc[df["cdr2_seq_aa_q_trim"].apply(lambda cdr: _valid_cdr3(cdr))]
    df = df.loc[df["cdr3_seq_aa_q_trim"].apply(lambda cdr: _valid_cdr3(cdr))]

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
