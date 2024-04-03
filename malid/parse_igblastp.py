import pandas as pd
from Bio import SeqIO
from io import StringIO
from typing import Dict

import logging

logger = logging.getLogger(__name__)


def _process_single(result: str, query_sequences: Dict[str, str]):
    start_index = 0
    while True:
        try:
            # Get the start and end indices of the alignment summary section.
            alignment_start_index = result.index(
                "# Alignment summary between query and top germline V gene hit",
                start_index,
            )
            alignment_end_index = result.index("Total\t", alignment_start_index)

            # Get the start index of the hit table section.
            hit_table_start_index = result.index(
                "# Hit table (the first field indicates the chain type of the hit)",
                start_index,
            )
        except ValueError as e:
            # If sections are not found, raise an error.
            raise ValueError("Required sections for parsing are missing.") from e

        try:
            # Now, find the index of the next "# Query" after the hit_table_start_index.
            is_there_more = result.index("# Query", hit_table_start_index)
        except ValueError:
            # If there's no further "# Query", we are at the end of the results.
            is_there_more = None

        # Read the alignment summary
        df = pd.read_csv(
            StringIO(result[alignment_start_index:alignment_end_index]),
            sep="\t",
            comment="#",
            header=None,
            names=[
                "region",
                "from",
                "to",
                "length",
                "matches",
                "mismatches",
                "gaps",
                "percent identity",
            ],
        ).set_index("region")

        # Compute the global percent identity
        df["weighted_identity"] = df["length"] * df["percent identity"]
        total_length = df["length"].sum()
        global_percent_identity = df["weighted_identity"].sum() / total_length

        # Read the V gene hit information.
        # If is_there_more is not None, then it is guaranteed to be after the hit_table_start_index.
        v_gene_data = result[
            hit_table_start_index : is_there_more if is_there_more is not None else None
        ]
        v_gene = pd.read_csv(
            StringIO(v_gene_data),
            sep="\t",
            comment="#",
            header=None,
            names=["type", "query_id", "v_gene"],
        )

        # Process the V gene hit information.
        v_gene = v_gene[v_gene["type"] == "V"]

        if v_gene.shape[0] == 1:
            match_details = v_gene.iloc[0]
            # Look up the query sequence
            query_sequence = query_sequences.get(match_details["query_id"], None)
            if query_sequence is not None:
                query_sequence = str(query_sequence.seq)
                sequence_parts = df.apply(
                    lambda row: query_sequence[int(row["from"] - 1) : int(row["to"])],
                    axis=1,
                )
                # Add the global percent identity to match details
                match_dict = match_details.to_dict()
                match_dict["global_percent_identity"] = global_percent_identity
                # Emit this match (merge the dictionaries)
                yield match_dict | sequence_parts.to_dict()

        # Update start_index for the next iteration to be the position where the next "# Query" was found.
        if is_there_more is not None:
            start_index = is_there_more
        else:
            break


def process(file_handle_fasta, file_handle_parse) -> pd.DataFrame:
    fasta_records = SeqIO.to_dict(SeqIO.parse(file_handle_fasta, "fasta"))
    results = pd.DataFrame(_process_single(file_handle_parse.read(), fasta_records))
    return results
