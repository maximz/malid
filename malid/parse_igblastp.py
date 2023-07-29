import pandas as pd
from Bio import SeqIO
from io import StringIO
from typing import Dict

import logging

logger = logging.getLogger(__name__)


def _process_single(result: str, query_sequences: Dict[str, str]):
    while True:
        try:
            is_there_more = result.index("# Query", 1)
        except ValueError:
            is_there_more = None

        df = pd.read_csv(
            StringIO(
                result[
                    result.index(
                        "# Alignment summary between query and top germline V gene hit"
                    ) : result.index("Total\t")
                ]
            ),
            sep="\t",
            comment="#",
            header=None,
            names=[
                "from",
                "to",
                "length",
                "matches",
                "mismatches",
                "gaps",
                "percent identity",
            ],
        )

        v_gene = pd.read_csv(
            StringIO(
                result[
                    result.index(
                        "# Hit table (the first field indicates the chain type of the hit)"
                    ) : is_there_more
                ]
            ),
            sep="\t",
            comment="#",
            header=None,
            names=["type", "query_id", "v_gene"],
        )

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
                # emit this match
                yield match_details.to_dict() | sequence_parts.to_dict()

        if is_there_more is not None:
            result = result[is_there_more:]
        else:
            break


def process(file_handle_fasta, file_handle_parse) -> pd.DataFrame:
    fasta_records = SeqIO.to_dict(SeqIO.parse(file_handle_fasta, "fasta"))
    results = pd.DataFrame(_process_single(file_handle_parse.read(), fasta_records))
    return results
