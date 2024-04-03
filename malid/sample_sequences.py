import collections
import itertools
import logging
from typing import Dict, List, Union
import pandas as pd
from malid.datamodels import GeneLocus
from malid import helpers

logger = logging.getLogger(__name__)

REQUIRED_CLONE_COUNTS_BY_ISOTYPE = {
    "IGHG": 100,
    "IGHA": 100,
    "IGHD-M": 500,
    "TCRB": 500,
}


def sample_sequences(
    df: pd.DataFrame, required_gene_loci: Union[GeneLocus, Dict[str, GeneLocus]]
) -> pd.DataFrame:
    """Sample sequences from a single specimen df.
    Parameters:
    - df must contain a single participant_label and specimen_label (no validation performed)
    - required_gene_loci:
        Determines which isotypes must be present in a specimen for it to be kept.
        Either a GeneLocus (single or composite), or a dict mapping specimen_label to single/composite GeneLocus.
    """
    specimen_label = df["specimen_label"].iloc[0]
    participant_label = df["participant_label"].iloc[0]

    # Remove short CDR3s
    df = df.loc[df["cdr3_aa_sequence_trim_len"] >= 8]

    if isinstance(required_gene_loci, collections.Mapping):
        # required_gene_loci is a dict of gene_loci, keyed by specimen_label
        required_gene_loci = required_gene_loci[specimen_label]
    GeneLocus.validate(required_gene_loci)

    # Get required_isotype_groups: all these isotypes must be present in a specimen for it to be kept.
    required_isotype_groups: List[str] = list(
        itertools.chain.from_iterable(
            [
                helpers.isotype_groups_kept[gene_locus]
                for gene_locus in required_gene_loci
            ]
        )
    )

    # Filter out certain isotype groups
    df = df.loc[df["isotype_supergroup"].isin(required_isotype_groups)]

    if df.shape[0] == 0:
        logger.info(
            f"Participant {participant_label} specimen {specimen_label} is empty"
        )
        return pd.DataFrame()  # or: return df.head(0)

    # Count number of clones per isotype
    clone_count_by_isotype = (
        df.groupby("isotype_supergroup", observed=True)["igh_or_tcrb_clone_id"]
        .nunique()
        .reindex(required_isotype_groups)
        .fillna(0)
    )

    # Ignore this sample entirely if not enough clones left in class-switched or naive isotypes
    # (Do this before filtering naive isotypes to SHM+ below. i.e. apply the filter to total IgM+D, not mutated subset only)
    # This check depends on which gene loci are expected/required
    if (
        (
            GeneLocus.BCR in required_gene_loci
            and clone_count_by_isotype.loc["IGHG"]
            < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["IGHG"]
        )
        or (
            GeneLocus.BCR in required_gene_loci
            and clone_count_by_isotype.loc["IGHA"]
            < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["IGHA"]
        )
        or (
            GeneLocus.BCR in required_gene_loci
            and clone_count_by_isotype.loc["IGHD-M"]
            < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["IGHD-M"]
        )
        or (
            GeneLocus.TCR in required_gene_loci
            and clone_count_by_isotype.loc["TCRB"]
            < REQUIRED_CLONE_COUNTS_BY_ISOTYPE["TCRB"]
        )
    ):
        # return blank
        # this should eliminate the partition once it's been read back in. otherwise try https://stackoverflow.com/a/50613803/130164
        logger.info(
            f"Removing {participant_label} specimen {specimen_label} because it did not have enough clones. Clone count by isotype: {clone_count_by_isotype.to_dict()}"
        )
        return pd.DataFrame()  # or: return df.head(0)

    # Filter out naive IgM/IgD with low SHM, but leave non-class-switched antigen-experienced cells
    df = df.loc[~((df["isotype_supergroup"] == "IGHD-M") & (df["v_mut"] < 0.01))]

    # Ignore entirely if fewer than 1000 sequences left. Something's wrong with this sample if so.
    if df.shape[0] < 1000:
        # return blank
        # this should eliminate the partition once it's been read back in. otherwise try https://stackoverflow.com/a/50613803/130164
        logger.info(
            f"Removing {participant_label} specimen {specimen_label} because it had fewer than 1000 sequences: {df.shape[0]}"
        )
        return pd.DataFrame()  # or: return df.head(0)

    # Sample one sequence per clone, per isotype-supergroup, per amplification, of a specimen
    # Clarification on the amplification_label, which can subdivide a specimen:
    # We may have a specimen processed multiple times from the same biological sample. This can allow evaluation of batch effects, for example. So we will sample from all copies of this specimen separately.
    # For example, our healthy controls in M64 were processed twice for cDNA, with two amplifications: M66-M64-cDNA amd M477-M64-cDNA.

    # However, there's also the concept of the replicate_label. This is for splitting a sample into IgM, IgG, IgA, etc., for example. Different replicate labels from the same amplification get combined/merged, effectively.

    # Clone IDs were created *across* all sequences for a single person (i.e. all specimens from this participant), so we don't need to change them when consolidating specimen fractions.

    # So to summarize, each amplification_label within a specimen_label is preserved in the sampling, but all replicate_labels from each amplification_label are merged together.
    grouping_keys = [
        "specimen_label",
        "amplification_label",
        "igh_or_tcrb_clone_id",
        "isotype_supergroup",
    ]
    # to be safe before using idxmax, we first need to make sure index is entirely unique:
    df = df.reset_index(drop=True)
    grpby = df.groupby(grouping_keys, observed=True, sort=False)
    # choose first sequence
    # df = grpby.head(n=1)
    # choose sequence with highest number of reads
    # (https://stackoverflow.com/a/22940775/130164)
    df = df.loc[grpby["num_reads"].idxmax(), :]
    # Count total number of reads for the whole clone
    df = pd.merge(
        df,
        grpby["num_reads"].sum().rename("total_clone_num_reads"),
        how="inner",
        validate="1:1",
        right_index=True,
        left_on=grouping_keys,
    )
    # Count number of clone members (number of unique VDJ sequences)
    df = pd.merge(
        df,
        grpby.size().rename("num_clone_members"),
        how="inner",
        validate="1:1",
        right_index=True,
        left_on=grouping_keys,
    )

    # Remove specimen if missing some isotypes
    if set(df["isotype_supergroup"].unique()) != set(required_isotype_groups):
        # return blank
        logger.info(
            f"Removing {participant_label} specimen {specimen_label} because it did not have all isotype groups: {df['isotype_supergroup'].unique()} instead of {required_isotype_groups}"
        )
        return pd.DataFrame()  # or: return df.head(0)

    # Remove remaining empty isotype groups from categorical column (cast defensively to categorical)
    df["isotype_supergroup"] = (
        df["isotype_supergroup"].astype("category").cat.remove_unused_categories()
    )

    return df
