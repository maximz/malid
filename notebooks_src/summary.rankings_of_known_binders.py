# %%
from malid import config
from summarynb import show
from IPython.display import display, Markdown

# %%

# %%
for gene_locus in config.gene_loci_used:
    display(Markdown(f"## {gene_locus}"))
    output_dir = config.paths.model_interpretations_output_dir / gene_locus.name

    for model in [3, 2]:
        display(Markdown(f"### Model {model}"))
        show(
            [
                [
                    output_dir
                    / f"known_binders_vs_healthy_controls.model{model}_rank_boxplot.fold_{fold_id}.png"
                    for fold_id in config.cross_validation_fold_ids
                ],
                [
                    output_dir
                    / f"known_binders_vs_healthy_controls.model{model}_rank_report.fold_{fold_id}.txt"
                    for fold_id in config.cross_validation_fold_ids
                ],
            ],
            headers=[
                f"{gene_locus}, fold {fold_id}"
                for fold_id in config.cross_validation_fold_ids
            ],
            max_width=400,
        )

    display(Markdown("---"))

# %%
