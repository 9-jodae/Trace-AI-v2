"""안전장치 자체가 동작하는지 확인한다."""

import os
import socket

import pytest
import safety

from app.core import config
from app.jev import client as jev_client
from app.llm import client as llm_client
from app.retrieval import embedding


def test_keys_are_fake():
    assert config.jev_api_key() == "fake-jev-key"
    assert config.embedding_api_key() == "fake-embedding-key"
    assert config.llm_api_key() == "fake-llm-key"
    for name in safety.PROVIDER_ENV:
        assert name not in os.environ


@pytest.mark.parametrize("host", ["api.typesafe.ai", "api.openai.com", "generativelanguage.googleapis.com"])
def test_external_network_is_blocked(host):
    before = len(safety.blocked_attempts)
    with pytest.raises(OSError):
        socket.create_connection((host, 443), timeout=1)
    del safety.blocked_attempts[before:]


def test_real_clients_are_forbidden():
    for factory in (jev_client.AsyncTypeSafeClient, llm_client.init_chat_model, embedding.init_embeddings):
        with pytest.raises(AssertionError):
            factory()
