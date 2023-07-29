"""
Find CDR1 and CDR2 annotations for all TCRB V genes.
Make sure to run `pyir setup` before running this script.
"""

import pandas as pd
from malid import config, get_v_sequence

df = get_v_sequence.get_tcrb_v_gene_annotations()

# Ignore rows when expected_v_call != v_call. Not sure if we should trust CDR1 and CDR2 then.
df = df.loc[df["expected_v_call"] == df["v_call"]]

# Pull out allele.
splits = df["v_call"].str.split("*", expand=True)
if splits.shape[1] != 2:
    raise ValueError(
        "Could not extract allele (v_call does not always have two parts)."
    )
splits.columns = ["v_gene", "v_allele"]
df = pd.concat([df, splits], axis=1)  # Add allele to df

# Confirm each gene's dominant allele (01) is included
for v_gene, grp in df.groupby("v_gene"):
    if "01" not in grp["v_allele"].values:
        raise ValueError(f"{v_gene} does not have a dominant 01 allele.")

if (
    (df["cdr1_aa"] == "").any()
    or (df["cdr2_aa"] == "").any()
    or df["cdr1_aa"].isna().any()
    or df["cdr2_aa"].isna().any()
):
    raise ValueError("Some CDR1 or CDR2 sequences were not extracted.")

fname_out = config.paths.metadata_dir / "tcrb_v_gene_cdrs.generated.tsv"
df.drop("expected_v_call", axis=1).to_csv(fname_out, sep="\t", index=None)
print(f"Wrote {fname_out}")
