"""
Convert our internal "part table" data format to AIRR TSV format.

We validated the outputs with:
    pip install airr
    airr-tools validate rearrangement -a converted_part_table.tsv
"""


import pandas as pd
from malid import etl
import typer

app = typer.Typer()

col_mapping = {
    "specimen_label": "repertoire_id",
    "amplification_locus": "locus",
    "isosubtype": "c_call",
    "trimmed_sequence": "sequence",
    "igh_clone_id": "clone_id",
    "tcrb_clone_id": "clone_id",
    "v_segment": "v_call",
    "d_segment": "d_call",
    "j_segment": "j_call",
    "v_score": "v_score",
    "d_score": "d_score",
    "j_score": "j_score",
    "stop_codon": "stop_codon",
    "v_j_in_frame": "vj_in_frame",
    "productive": "productive",
    "fr1_seq_nt_q": "fwr1",
    "cdr1_seq_nt_q": "cdr1",
    "fr2_seq_nt_q": "fwr2",
    "cdr2_seq_nt_q": "cdr2",
    "fr3_seq_nt_q": "fwr3",
    "cdr3_seq_nt_q": "cdr3",
    "post_seq_nt_q": "fwr4",
    "fr1_seq_aa_q": "fwr1_aa",
    "cdr1_seq_aa_q": "cdr1_aa",
    "fr2_seq_aa_q": "fwr2_aa",
    "cdr2_seq_aa_q": "cdr2_aa",
    "fr3_seq_aa_q": "fwr3_aa",
    "cdr3_seq_aa_q": "cdr3_aa",
    "post_seq_aa_q": "fwr4_aa",
    "v_sequence": "v_sequence_alignment",
    "d_sequence": "d_sequence_alignment",
    "j_sequence": "j_sequence_alignment",
}

additional_columns = [
    "rev_comp",
    "sequence_alignment",
    "germline_alignment",
    "junction",
    "junction_aa",
    "v_cigar",
    "d_cigar",
    "j_cigar",
]


@app.command()
def plot(
    input: str,
    output: str,
):
    """Convert our internal "part table" data format to AIRR TSV format."""
    # Import
    df = pd.read_csv(
        input,
        sep="\t",
        index_col=0,
    )

    # Trim AA strings
    for aa_col in [
        "fr1_seq_aa_q",
        "cdr1_seq_aa_q",
        "fr2_seq_aa_q",
        "cdr2_seq_aa_q",
        "fr3_seq_aa_q",
        "cdr3_seq_aa_q",
        "post_seq_aa_q",
    ]:
        # TCR will have empty FR1, CDR1, FR2, and CDR2. (In etl.py, we deterministically fill in FR1 through FR3.)
        # So we will skip any empty columns, for which the .str accessor calls in _trim_sequences would be illegal.
        if df[aa_col].notna().any():
            df[aa_col] = etl._trim_sequences(df[aa_col])

    # Combine run_label and trimmed_read_id into a sequence_id
    df["sequence_id"] = df["run_label"] + "|" + df["trimmed_read_id"].astype(str)

    # Subset columns
    desired_columns = ["sequence_id"] + list(col_mapping.keys())
    # Some columns may be BCR only or TCR only, so don't raise an error if they're missing
    desired_columns = [c for c in desired_columns if c in df.columns]
    df = df[desired_columns]

    # Rename columns
    df = df.rename(
        columns=col_mapping,
        # Some columns may be BCR only or TCR only, so don't raise an error if they're missing
        errors="ignore",
    )

    # Add additional columns
    for col in additional_columns:
        df[col] = pd.Series(dtype="object")  # specify dtype to avoid warning

    # Export
    df.to_csv(
        output,
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    app()
