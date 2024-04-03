import numpy as np
import pandas as pd
import logging
import pytest

from malid.datamodels import GeneLocus, healthy_label
from malid import config
from malid.etl import (
    dtypes_read_in,
    dtypes_expected_after_preprocessing,
    preprocess_each_participant_table,
    _trim_sequences,
    _split_post_seq_into_cdr3_and_fwr4,
    load_participant_data_external,
    read_boydlab_participant_table,
)


def test_etl_with_empty_sample_dataframe():
    """Confirm that ETL works with an empty dataframe, and provides the expected empty result dataframe with appropriate (blank) columns."""

    # Note that this isn't the full list of metadata_whitelist columns we expect, but it's enough to test the logic.
    metadata_whitelist = pd.DataFrame(
        {"specimen_label": ["specimen1"], "participant_label": ["participant1"]}
    )

    # The preprocess_each_participant_table logic depends on gene locus, so try all.
    for gene_locus in GeneLocus:
        orig = pd.DataFrame(
            {c: pd.Series(dtype=t) for c, t in dtypes_read_in[gene_locus].items()}
        )
        processed = preprocess_each_participant_table(
            df=orig,
            gene_locus=gene_locus,
            metadata_whitelist=metadata_whitelist,
        )
        assert processed.shape[0] == 0
        assert processed.shape[1] > 5
        # Check final column names against expected final dtypes
        assert set(processed.columns) == set(dtypes_expected_after_preprocessing.keys())


# When we add new data formats, we should include a representative example in the automated tests below.


@pytest.mark.parametrize("gene_locus", GeneLocus)
def test_boydlab_etl(gene_locus: GeneLocus):
    # must match what's in the files
    metadata_whitelist = pd.concat(
        [
            pd.DataFrame(
                {"specimen_label": ["M01-S001"], "participant_label": ["BFI-0000001"]}
            ),
            pd.DataFrame(
                {
                    "specimen_label": ["M01-S002"],
                    "participant_label": ["BFI-0000002"],
                    "disease": healthy_label,
                    "disease_subtype": healthy_label,
                }
            ),
            pd.DataFrame(
                {
                    "specimen_label": ["M01-S003", "M01-S003_CD4", "M01-S003_CD8"],
                    "specimen_label_override": ["M01-S003", "M01-S003", "M01-S003"],
                    "amplification_label_override": [
                        "M01-S003",
                        "M01-S003",
                        "M01-S003",
                    ],
                    "replicate_label_override": [
                        "M01-S003",
                        "M01-S003_CD4",
                        "M01-S003_CD8",
                    ],
                    "cell_type": [np.nan, "CD4 T", "CD8 T"],
                    "participant_label": "BFI-0000003",
                    "disease": "DiseaseA",
                    "disease_subtype": "DiseaseA - Flare",
                }
            ),
        ],
        axis=0,
    ).reset_index(drop=True)

    df = preprocess_each_participant_table(
        df=read_boydlab_participant_table(
            fname=config.paths.tests_snapshot_dir
            / f"example_part_table.{gene_locus.name.lower()}.tsv.bz2",
            gene_locus=gene_locus,
        ),
        gene_locus=gene_locus,
        metadata_whitelist=metadata_whitelist,
    )
    assert df.shape[0] > 1
    assert set(df.columns) == set(dtypes_expected_after_preprocessing.keys())
    assert (
        not df["cdr3_seq_aa_q_trim"].str.startswith("C").any()
    ), "CDR3s should be trimmed"

    # example_part_table2: Multiple amplifiations per sample.
    df = preprocess_each_participant_table(
        df=read_boydlab_participant_table(
            fname=config.paths.tests_snapshot_dir
            / f"example_part_table2.{gene_locus.name.lower()}.tsv.bz2",
            gene_locus=gene_locus,
        ),
        gene_locus=gene_locus,
        metadata_whitelist=metadata_whitelist,
    )
    assert df.shape[0] > 1
    assert set(df.columns) == set(dtypes_expected_after_preprocessing.keys())
    assert (
        not df["cdr3_seq_aa_q_trim"].str.startswith("C").any()
    ), "CDR3s should be trimmed"
    assert (df["specimen_label"] == "M01-S002").all()
    if gene_locus == GeneLocus.BCR:
        assert np.array_equal(
            df["amplification_label"].unique(),
            ["M01-S002_cDNA_PCR", "M03-M01-S002_cDNA_PCR"],
        )
        assert df["replicate_label"].nunique() > 2
    elif gene_locus == GeneLocus.TCR:
        # TODO: Why is this?
        assert np.array_equal(
            df["amplification_label"].unique(),
            ["M01-S002_cDNA_TCRB", "M03-M01-S002_cDNA_PCR_TCRB"],
        )
        assert np.array_equal(
            df["replicate_label"].unique(),
            ["M01-S002_cDNA_TCRB_R1", "M03-M01-S002_cDNA_PCR_TCRB_R1"],
        )

    # example_part_table3:
    # Samples with "_CD8" and "_CD4" cell fraction suffixes
    # Confirm they are removed, with cell_type column set to "CD8 T" and "CD4 T" respectively.
    df = preprocess_each_participant_table(
        df=read_boydlab_participant_table(
            fname=config.paths.tests_snapshot_dir
            / f"example_part_table3.{gene_locus.name.lower()}.tsv.bz2",
            gene_locus=gene_locus,
        ),
        gene_locus=gene_locus,
        metadata_whitelist=metadata_whitelist,
    )
    assert df.shape[0] > 1
    assert set(df.columns) == set(dtypes_expected_after_preprocessing.keys())
    assert (
        not df["cdr3_seq_aa_q_trim"].str.startswith("C").any()
    ), "CDR3s should be trimmed"
    assert (df["specimen_label"] == "M01-S003").all()
    assert (df["disease"] == "DiseaseA").all()
    assert (df["disease_subtype"] == "DiseaseA - Flare").all()
    if gene_locus == GeneLocus.BCR:
        assert df["cell_type"].isna().all()
        assert (df["amplification_label"] == "M01-S003").all()
        assert (df["replicate_label"] == "M01-S003").all()
    elif gene_locus == GeneLocus.TCR:
        assert np.array_equal(df["cell_type"], ["CD4 T"] * 5 + ["CD8 T"] * 4)
        assert (df["amplification_label"] == "M01-S003").all()
        assert np.array_equal(
            df["replicate_label"].unique(), ["M01-S003_CD4", "M01-S003_CD8"]
        )


def test_adaptive_etl():  # (caplog: pytest.LogCaptureFixture): # see caplog comment below
    base_path = (
        config.paths.tests_snapshot_dir / "sample_adaptive_immuneaccess_format_data"
    )
    # Participant 1 (two samples) comes from ImmuneAccess format
    # Participants 2-4 (one sample each) come from Emerson format
    # Participant 5 (one sample) comes from ImmuneCode format
    participant_samples_all = pd.DataFrame(
        {
            "study_name": [
                "sample_study",
                "sample_study",
                "sample_emerson_format",
                "sample_emerson_format",
                "sample_emerson_format",
                "sample_immunecode_format",
            ],
            "participant_label": [
                "participant1",
                "participant1",
                "participant2",
                "participant3",
                "participant4",
                "participant5",
            ],
            # specimen_label: globally unique, but may have several amplifications and replicates.
            "specimen_label": [
                "participant1_specimen1",
                "participant1_specimen2",
                "participant2_specimen1",
                "participant3_specimen2",
                "participant4_specimen3",
                "participant5_specimen1",
            ],
            # amplification_label: globally unique, but may have several replicates.
            "amplification_label": [
                "participant1_specimen1",
                "participant1_specimen2",
                "participant2_specimen1",
                "participant3_specimen2",
                "participant4_specimen3",
                "participant5_specimen1",
            ],
            # replicate_label: globally unique.
            "replicate_label": [
                "participant1_specimen1",
                "participant1_specimen2",
                "participant2_specimen1",
                "participant3_specimen2",
                "participant4_specimen3",
                "participant5_specimen1",
            ],
            # sample_name: not globally unique, but should be unique within each study. used in the fasta header and igblast parsed "id" column.
            "sample_name": [
                "sample1",
                "sample2_empty",
                "specimen1",
                "specimen2",
                "specimen3",
                "specimen1",
            ],
        }
    )

    # with caplog.at_level(logging.WARNING, logger="malid.etl"):
    # Use caplog to capture logged warnings, which are expected for sample2_empty.
    # However, caplog appears to be flaky - which others have also reported when running on full test suite (but not individual tests).
    # We confirmed the root, tests.test_etl, malid, and malid.etl loggers are all set to propagate; that's not the issue.
    # We also tried this alternative, but no dice: https://stackoverflow.com/a/71913487/130164
    # Decision: skip this check. Leaving as comments in case we find a way to bring this back.
    for (
        participant_label,
        single_participant_samples,
    ) in participant_samples_all.groupby("participant_label"):
        df = load_participant_data_external(
            participant_samples=single_participant_samples,
            gene_locus=GeneLocus.TCR,
            base_path=base_path,
            is_adaptive=True,
        )
        assert df.shape[0] >= 1
        assert set(df.columns) == set(dtypes_expected_after_preprocessing.keys())
        assert (
            not df["cdr3_seq_aa_q_trim"].str.startswith("C").any()
        ), "CDR3s should be trimmed"

    # assert (
    #     "Skipping empty sample sample2_empty in study sample_study" in caplog.text
    # ), caplog.text


def test_external_etl():  # follows same pattern as test_adaptive_etl
    base_path = config.paths.tests_snapshot_dir / "sample_external_format_data"
    # Participant 1 (one sample) comes from Chudakov format
    participant_samples_all = pd.DataFrame(
        {
            "study_name": [
                # Note that Britanova is a special-cased study name that triggers unique behavior.
                "Britanova",
                "sample_kim_covid_format",
                "sample_shomuradova_format",
                "sample_briney_format",
            ],
            "participant_label": [
                "participant1",
                "participant2",
                "participant3",
                "participant4",
            ],
            # specimen_label: globally unique, but may have several amplifications and replicates.
            "specimen_label": [
                "participant1_specimen1",
                "participant2_specimen1",
                "participant3_specimen1",
                "participant4_specimen1",
            ],
            # amplification_label: globally unique, but may have several replicates.
            "amplification_label": [
                "participant1_specimen1",
                "participant2_specimen1",
                "participant3_specimen1",
                "participant4_specimen1",
            ],
            # replicate_label: globally unique.
            "replicate_label": [
                "participant1_specimen1",
                "participant2_specimen1",
                "participant3_specimen1",
                "participant4_specimen1",
            ],
            # sample_name: not globally unique, but should be unique within each study. used in the fasta header and igblast parsed "id" column.
            "sample_name": [
                "sample1",
                "specimenA_d1",
                "specimen1",
                "specimen1_1",
            ],
            #
            ### The following columns are not passed in dataframe along to inner load_participant_data_external function, but used to configure parameters of the load_participant_data_external function itself.
            # gene_locus:
            "gene_locus": [
                GeneLocus.TCR,
                GeneLocus.BCR,
                GeneLocus.TCR,
                GeneLocus.BCR,
            ],
            # Allow some studies to be exempted from read count column requirements
            "expect_a_read_count_column": [
                True,
                False,  # No read counts in Kim iReceptor Covid
                False,  # No read counts in Shomuradova
                False,  # No read counts in Briney
            ],
            # Allow custom file extensions for some studies. Default is tsv
            "file_extension": [
                "tsv",
                "tsv",
                "tsv",
                "csv",
            ],
        }
    )

    for (
        (participant_label, gene_locus),
        single_participant_samples,
    ) in participant_samples_all.groupby(
        ["participant_label", "gene_locus"], observed=True, sort=False
    ):
        expect_a_read_count_column = single_participant_samples[
            "expect_a_read_count_column"
        ].iloc[0]
        assert (
            single_participant_samples["expect_a_read_count_column"]
            == expect_a_read_count_column
        ).all()

        file_extension = single_participant_samples["file_extension"].iloc[0]
        assert (single_participant_samples["file_extension"] == file_extension).all()

        df = load_participant_data_external(
            participant_samples=single_participant_samples,
            gene_locus=gene_locus,
            base_path=base_path,
            is_adaptive=False,
            expect_a_read_count_column=expect_a_read_count_column,
            file_extension=file_extension,
        )
        assert df.shape[0] >= 1
        assert set(df.columns) == set(dtypes_expected_after_preprocessing.keys())
        assert (
            not df["cdr3_seq_aa_q_trim"].str.startswith("C").any()
        ), "CDR3s should be trimmed"


def test_trim_sequences():
    trimmed = _trim_sequences(pd.Series(["V Y Y T G S T", " N A ", "    ", " ... "]))
    # complex series equals comparison to allow nan==nan
    assert trimmed.equals(pd.Series(["VYYTGST", "NA", np.nan, np.nan]))


def test_split_post_seq_into_cdr3_and_fwr4():
    df = pd.DataFrame(
        {
            # combined sequence:
            "post_seq_aa_q": [
                "AS PMVHGYTFGS",
                "ASPMVHGYTFGS",
                "BSPMKGHTFGH",
                "CYTMAKLOPQ",
                np.nan,
                "KLOPQ",
            ],
            # prefix sequence:
            "cdr3_seq_aa_q": [
                "ASPMVHGYT",
                "BSPMK",
                "B   S     PMK",
                "CYTM",
                "CYTM",
                "CYTM",
            ]
            # the spaces get trimmed
        },
        index=["row1", "row2", "row3", "row4", "row5", "row6"],
    )
    result = _split_post_seq_into_cdr3_and_fwr4(df, "cdr3_seq_aa_q", "post_seq_aa_q")
    assert result.equals(
        pd.Series(
            ["FGS", "ASPMVHGYTFGS", "GHTFGH", "AKLOPQ", np.nan, "KLOPQ"], index=df.index
        )
    )
