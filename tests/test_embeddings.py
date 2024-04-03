#!/usr/bin/env python

from typing import Union, Type
import pytest
import numpy as np
import pandas as pd

from malid import embedders
from malid.datamodels import GeneLocus
from malid.embedders.ablang import AbLangEmbeddder
from malid.embedders.base_embedder import BaseEmbedder, BaseFineTunedEmbedder


@pytest.fixture
def sequences() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "fr1_seq_aa_q_trim": "AAS",
                "cdr1_seq_aa_q_trim": "GFTFSSYS",
                "fr2_seq_aa_q_trim": "MNWVRQAPGKGLEWVSV",
                "cdr2_seq_aa_q_trim": "IYSGGGT",
                "fr3_seq_aa_q_trim": "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
                "cdr3_seq_aa_q_trim": "ARADRGYPDY",
                "post_seq_aa_q_trim": "WGQGTLVTVSS",
            }
        ]
        * 3
    )


@pytest.fixture
def sequences_batch(sequences) -> pd.DataFrame:
    return pd.concat([sequences] * 51, axis=0)


def test_embedders_dont_have_duplicate_names():
    # Confirm all names are unique.
    assert len(embedders._EMBEDDERS_DICT) == len(embedders._EMBEDDERS)


def test_embedders_dont_have_duplicate_friendly_names():
    # Confirm all friendly names are unique.
    assert len(set(e.friendly_name for e in embedders._EMBEDDERS)) == len(
        embedders._EMBEDDERS
    )


@pytest.mark.xfail(raises=KeyError)
def test_embedder_get_invalid_name():
    embedders.get_embedder_by_name("non_existent")


@pytest.mark.parametrize(
    "embedder_name,embedder_class", embedders._EMBEDDERS_DICT.items()
)
def test_embedder(
    sequences,
    embedder_name: str,
    embedder_class: Union[Type[BaseEmbedder], Type[BaseFineTunedEmbedder]],
    mocker,
):
    assert embedder_name == embedder_class.name  # sanity check the name field matches
    assert embedder_class.friendly_name is not None  # confirm the friendly name is set

    # If fine tuned, fake it: pretend as if we are loading fine tuned parameters
    if issubclass(embedder_class, BaseFineTunedEmbedder):
        mocker.patch.object(embedder_class, "_load_fine_tuned_model")

    embedder_instance = embedder_class(fold_id=0, gene_locus=GeneLocus.BCR)
    # sanity checks
    assert embedder_name == embedder_instance.name
    assert embedder_class.friendly_name == embedder_instance.friendly_name

    # Confirm that if fine tuned, the constructor called _load_fine_tuned_model (which we've mocked out)
    if isinstance(embedder_instance, BaseFineTunedEmbedder):
        embedder_instance._load_fine_tuned_model.assert_called_once_with(
            0, GeneLocus.BCR
        )

    # Special case: AbLangEmbedder does not yet implement calculate_cross_entropy_loss
    supports_cross_entropy_loss = not issubclass(embedder_class, AbLangEmbeddder)

    # test the is_fine_tuned property
    assert (
        embedder_class.is_fine_tuned
        == embedder_instance.is_fine_tuned
        == issubclass(embedder_class, BaseFineTunedEmbedder)
        == isinstance(embedder_instance, BaseFineTunedEmbedder)
        == ("fine_tuned" in embedder_class.name)
        == ("off the shelf" not in embedder_class.friendly_name)
    )

    if issubclass(embedder_class, BaseFineTunedEmbedder):
        # Test that we can recover the non fined tune version.
        non_fine_tuned_version: Type[
            BaseEmbedder
        ] = embedder_class.non_fine_tuned_version
        assert (
            non_fine_tuned_version is not None
        ), f"non_fine_tuned_version must be set for {embedder_class}"
        assert embedder_class is not non_fine_tuned_version  # They are not the same

        # Fine-tuned version is not necessarily a subclass of the non-fine-tuned version.
        # Don't do this:
        # assert issubclass(embedder_class, non_fine_tuned_version)

        # Non fine tuned version should be a subclass of BaseEmbedder, but not of BaseFineTunedEmbedder
        assert issubclass(non_fine_tuned_version, BaseEmbedder)
        assert not issubclass(non_fine_tuned_version, BaseFineTunedEmbedder)

        # Non fine tuned version should not be BaseEmbedder or BaseFineTunedEmbedder directly
        assert non_fine_tuned_version is not BaseEmbedder
        assert non_fine_tuned_version is not BaseFineTunedEmbedder

        # Important: when we convert to non-fine-tuned version, we should not silently switch the sequence region being embedded.
        assert (
            non_fine_tuned_version.embedder_sequence_content
            == embedder_class.embedder_sequence_content
        )
    else:
        # Test that there is no further non-fine-tuned version.
        assert embedder_class.non_fine_tuned_version is embedder_class

    embedded_sequences = embedder_instance.embed(sequences, dtype=np.float16)
    assert embedded_sequences.shape == (len(sequences), embedder_class.dimensionality)

    if supports_cross_entropy_loss:
        # Should support mixed-length sequences
        loss = embedder_instance.calculate_cross_entropy_loss(
            sequences=["CAR", "CAS", "CARS"]
        )
        assert type(loss) == float
        assert not np.isnan(loss)
