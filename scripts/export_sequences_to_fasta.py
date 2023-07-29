"""
Export sequence column from any CSV as a fasta file.
Fasta headers are >[name argument]|[row number]
"""

import numpy as np
import pandas as pd
import click


@click.command()
@click.option("--input", "fname_in", type=str, help="input filename", required=True)
@click.option("--output", "fname_out", type=str, help="output filename", required=True)
@click.option(
    "--name",
    "specimen_label",
    type=str,
    help="specimen label to use in fasta header",
    required=True,
)
@click.option(
    "--column",
    "seq_col_name",
    default="rearrangement",
    type=str,
    show_default=True,
    help="sequence column name to export",
)
@click.option(
    "--separator",
    default=",",
    type=str,
    show_default=True,
    help="CSV separator. To escape 'tab' on command line, use: --separator $'\\t'",
)
def run(
    fname_in,
    fname_out,
    specimen_label,
    seq_col_name="rearrangement",
    separator=",",
    # special N/A values for Adaptive data
    na_values=["no data", "unknown"],
):
    # read in
    df = pd.read_csv(fname_in, sep=separator, na_values=na_values)

    # drop N/A or empty string
    df[seq_col_name] = df[seq_col_name].mask(
        df[seq_col_name] == ""
    )  # change empty string to NaN
    df = df.dropna(subset=[seq_col_name])

    # Use numpy savetxt because pandas to_csv doesn't like custom format and to_string truncates long sequences
    np.savetxt(
        fname_out,
        df.apply(
            # id is: specimen label|row number
            lambda row: f"> {specimen_label}|{row.name}\n{row[seq_col_name]}",
            axis=1,
        ).values,
        fmt="%s",
    )


if __name__ == "__main__":
    run()
