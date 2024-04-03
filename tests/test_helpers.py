#!/usr/bin/env python
from malid import helpers
from malid.datamodels import GeneLocus, diseases
from genetools.palette import HueValueStyle


def test_disease_color_palette():
    # confirm all diseases are present in the map
    assert all(
        disease in helpers.disease_color_palette.keys() for disease in diseases
    ), "helpers.disease_color_palette is incomplete -- should match datamodels.diseases"
    # confirm all disease colors are unique
    assert len(set(helpers.disease_color_palette.values())) == len(
        helpers.disease_color_palette.values()
    )


def test_study_name_color_palette():
    # confirm all study names are present in the map
    all_specimens = helpers.get_all_specimen_info()
    all_specimens = all_specimens[all_specimens["in_training_set"]]
    expected_study_names = all_specimens["study_name"].unique()
    assert all(
        study_name in helpers.study_name_color_palette.keys()
        for study_name in expected_study_names
    ), f"helpers.study_name_color_palette is incomplete, expected {expected_study_names}"

    # confirm all study name colors are unique
    # first extract hex codes
    extracted_colors = HueValueStyle.huestyles_to_colors_dict(
        helpers.study_name_color_palette
    )
    assert len(set(extracted_colors.values())) == len(extracted_colors.values())


def test_gene_locus_listed():
    assert all(
        gene_locus in helpers.isotype_groups_kept.keys() for gene_locus in GeneLocus
    )


def test_enrich_metadata():
    # pretend we have all but one column
    partial = (
        helpers.get_all_specimen_info()
        .set_index("specimen_label")
        .drop(columns="age_group_pediatric")
    )
    enriched = helpers.enrich_metadata(partial)
    assert "age_group_pediatric" not in partial.columns
    # column has returned
    assert "age_group_pediatric" in enriched.columns
    assert partial.shape[0] == enriched.shape[0]
    assert partial.index.equals(enriched.index)
    assert partial.equals(enriched[partial.columns])

    # redo after enriched once: no more changes expected
    enriched_again = helpers.enrich_metadata(enriched)
    assert enriched.equals(enriched_again)


def test_v_gene_sort():
    data = [
        "IGHV1",
        "IGHV1-19",
        "IGHV1-29",
        "IGHV2",
        "IGHV10",
        "IGHV20",
        "IGHV5",
        "IGHV3-7",
        "IGHV3-11",
    ]
    sorted = helpers.v_gene_sort(data)
    assert sorted == [
        "IGHV1",
        "IGHV1-19",
        "IGHV1-29",
        "IGHV2",
        "IGHV3-7",
        "IGHV3-11",
        "IGHV5",
        "IGHV10",
        "IGHV20",
    ]
