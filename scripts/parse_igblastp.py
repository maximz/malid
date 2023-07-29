import click
from malid.parse_igblastp import process

import logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--fasta",
    type=str,
    required=True,
    help="Original fasta file",
)
@click.option(
    "--parse",
    type=str,
    required=True,
    help="IgBlast output file",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output file name",
)
@click.option(
    "--separator",
    default=",",
    type=str,
    show_default=True,
    help="Output file format separator. Defaults to comma (CSV). To escape 'tab' on command line, use: --separator $'\\t'",
)
def run(
    fasta: str,
    parse: str,
    output: str,
    separator: str,
):
    with open(fasta, "r") as file_handle_fasta:
        with open(parse, "r") as file_handle_parse:
            results = process(
                file_handle_fasta=file_handle_fasta, file_handle_parse=file_handle_parse
            )
            results.to_csv(output, index=None, sep=separator)


if __name__ == "__main__":
    run()
