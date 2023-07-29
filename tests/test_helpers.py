#!/usr/bin/env python

from malid import helpers
from malid.datamodels import GeneLocus
from genetools.palette import HueValueStyle


def test_disease_color_palette():
    # confirm all diseases are present in the map
    assert all(
        disease in helpers.disease_color_palette.keys() for disease in helpers.diseases
    ), "helpers.disease_color_palette is incomplete -- should match helpers.diseases"
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
