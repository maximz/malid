#!/usr/bin/env python

import os
import sys

from malid import config


def test_env_var_for_embedder(monkeypatch):
    assert config._default_embedder != "unirep"
    monkeypatch.setenv("EMBEDDER_TYPE", "unirep")
    assert config.choose_embedder().name == "unirep"


def test_embedder_has_name():
    # making sure that we can access name attribute without initializing the embedder
    assert config.embedder.name is not None


def test_env_vars():
    assert os.getenv("TRANSFORMERS_CACHE") is not None


def test_logging():
    # basic test of logging to make sure no recursive logger->stderr->logger issues
    config.logger.debug("test")
    config.logger.error("testerr")
    print("test2")
    sys.stderr.write("testerr2")
