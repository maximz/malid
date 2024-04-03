"""v_identity no longer available. Here is how we produce v_sequence in sort script"""

import pandas as pd

segment_prefixes = [
    "pre_seq_nt_",
    "fr1_seq_nt_",
    "cdr1_seq_nt_",
    "fr2_seq_nt_",
    "cdr2_seq_nt_",
    "fr3_seq_nt_",
    "cdr3_seq_nt_",
    "post_seq_nt_",
]


def left_right_mask(query, mask, blanks=" "):
    if len(mask) == 0:
        return None
    else:
        assert len(query) == len(mask)
        left_trim = len(mask) - len(mask.lstrip(blanks))
        right_trim = len(mask) - len(mask.rstrip(blanks))

        if left_trim == len(mask):
            return ""
        else:
            if right_trim == 0:
                return query[left_trim:]
            else:
                return query[left_trim:-right_trim]


def complete_sequences(df):
    """
    produces columns to be stored as df["v_sequence"], df["d_sequence"], df["j_sequence"]
    """
    full_sequences = {}
    for sequence_type in ["q", "v", "d", "j"]:
        col_names = [prefix + sequence_type for prefix in segment_prefixes]
        # sum these columns for each row (result is a series with shape df.shape[0])
        full_sequences[sequence_type] = df[col_names].fillna("").sum(axis=1)
    full_sequences = pd.DataFrame(full_sequences)
    if full_sequences.shape[0] != df.shape[0]:
        raise ValueError("shape error")

    # do this in one scan
    # https://stackoverflow.com/a/49192682/130164
    def get_full_sequences(row):
        return (
            left_right_mask(row["q"], row["v"]),
            left_right_mask(row["q"], row["d"]),
            left_right_mask(row["q"], row["j"]),
        )

    # df[['v_sequence', 'd_sequence', 'j_sequence']] = full_sequences.apply(get_full_sequences, axis=1, result_type="expand")
    # even faster: https://stackoverflow.com/a/48134659/130164 :
    return zip(*full_sequences.apply(get_full_sequences, axis=1))


def get_tcrb_v_gene_annotations() -> pd.DataFrame:
    """
    Find CDR1 and CDR2 annotations for all TCRB V genes:

    1. Get all TCRB V gene germline nucleotide sequences, the same way that PyIR does in its setup code (https://github.com/crowelab/PyIR/blob/3b07cbb1af0b17479d6c88974a681a04a7429b8d/pyir/data/bin/setup_germline_library.py).
    2. The dots represent gaps per the IMGT numbering scheme. As in PyIR, we remove them. So we should end up with an identical set of sequences to PyIR's set stored in `site-packages/crowelab_pyir/data/germlines/TCR/human/human_TCR_V.fasta` (although that includes TCR-A as well). PyIR then builds a Blast database from these sequences.
    3. Run PyIR's IgBlast against these sequences and extract CDR1+2.

    Make sure to run `pyir setup` before running this script.
    """

    import tempfile
    import requests
    import crowelab_pyir

    with tempfile.NamedTemporaryFile(mode="w") as f:
        original_fasta = requests.get(
            "https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/TR/TRBV.fasta",
            # skip SSL certificate verification
            verify=False,
        ).text
        f.write(original_fasta.replace(".", ""))  # replace the gaps
        f.flush()
        all_seqs_output = crowelab_pyir.PyIR(
            query=f.name,
            args=[
                "--outfmt",
                "dict",
                "--receptor",
                "TCR",
                "--species",
                "human",
                "--input_type",
                "fasta",
                "--sequence_type",
                "nucl",
                "--silent",
            ],
        ).run()
        if len(all_seqs_output.keys()) != original_fasta.count(">"):
            # Run PyIR without silent?
            # Use SeqIO to reveal which records were lost?
            raise ValueError("Some sequences were not IgBlasted.")
        return pd.DataFrame(
            [
                {
                    "expected_v_call": key.split("|")[1],
                    "v_call": result["v_call"].split(",")[0],
                    "fwr1_aa": result["fwr1_aa"],
                    "cdr1_aa": result["cdr1_aa"],
                    "fwr2_aa": result["fwr2_aa"],
                    "cdr2_aa": result["cdr2_aa"],
                    "fwr3_aa": result["fwr3_aa"],
                }
                for key, result in all_seqs_output.items()
            ]
        ).sort_values("expected_v_call")
