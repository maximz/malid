#!/usr/bin/env python

import pytest

from malid import embedders
from malid.embedders.unirep import UnirepEmbedder, UnirepFineTunedEmbedder


@pytest.fixture
def sequences():
    return ["ASDF", "YZKAL", "QQLAMEHALQP"]


@pytest.fixture
def sequences_batch(sequences):
    return sequences * 51


# TODO: speed up tests: initialize complex embedders in a fixture once, and reuse


def test_embedder_get_by_name():
    # use some global embedder type registry, or some kind of introspection, or __all__ to generate this list?
    for embedder in [UnirepEmbedder]:
        assert embedders.get_embedder_by_name(embedder.name) == embedder


@pytest.mark.xfail(raises=ValueError)
def test_embedder_get_invalid_name():
    embedders.get_embedder_by_name("non_existent")


def test_determine_if_fine_tuned_embedder():
    assert not UnirepEmbedder().is_fine_tuned
    assert UnirepFineTunedEmbedder().is_fine_tuned


def test_unirep(sequences):
    embedded = UnirepEmbedder().embed(sequences)
    assert embedded.shape == (len(sequences), 1900)
