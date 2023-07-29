import click

from malid import config
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from enumchoice import EnumChoice

## Common CLI options.

# More structured alternatives to consider:
# - https://stackoverflow.com/questions/71125871/create-structured-data-directly-from-python-click
# - https://stackoverflow.com/questions/56185880/commands-with-multiple-common-options-going-into-one-argument-using-custom-decor
# - https://stackoverflow.com/questions/50061342/is-it-possible-to-reuse-python-click-option-decorators-for-multiple-commands


accepts_gene_loci = click.option(
    "--locus",
    "gene_locus",
    multiple=True,
    default=[
        config.gene_loci_used
    ],  # this is a list containing a single multi-flag GeneLocus item!
    show_default=True,
    type=EnumChoice(GeneLocus, case_sensitive=False),
    help="Gene locus to train on",
)  # if multiple isn't True, default could be 'test' or Test.test


accepts_target_obs_columns = click.option(
    "--target_obs_column",
    multiple=True,
    default=config.classification_targets,
    show_default=True,
    type=EnumChoice(TargetObsColumnEnum, case_sensitive=False),
    help="Target observation columns to train on",
)  # if multiple isn't True, default could be 'test' or Test.test

accepts_fold_ids = click.option(
    "--fold_id",
    "fold_ids",
    multiple=True,
    default=config.all_fold_ids,
    type=int,
    show_default=True,
    help="Fold IDs to run.",
)

accepts_sample_weight_strategies = click.option(
    "--sample_weight_strategy",
    multiple=True,
    default=[SampleWeightStrategy.ISOTYPE_USAGE],  # , SampleWeightStrategy.NONE
    show_default=True,
    type=EnumChoice(SampleWeightStrategy, case_sensitive=False),
    help="Sample weight strategies to use",
)  # if multiple isn't True, default could be 'test' or Test.test

accepts_n_jobs = click.option(
    "--n_jobs",
    default=1,
    type=int,
    show_default=True,
    help="Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.",
)
