"""
Pytest configuration file

This file configures pytest to find the project modules.
"""

import sys
from importlib import import_module
from pathlib import Path
import socket

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_model: permits a test to call real model or network services",
    )


@pytest.fixture(autouse=True)
def block_external_services(monkeypatch, request):
    if request.node.get_closest_marker("real_model"):
        return

    def blocked_call(*args, **kwargs):
        raise RuntimeError(
            "Real model and network calls are disabled in unit tests; "
            "use an explicit real_model marker for an integration test."
        )

    import urllib.request

    def blocked_build_opener(*args, **kwargs):
        class BlockedOpener:
            open = staticmethod(blocked_call)

        return BlockedOpener()

    monkeypatch.setattr(urllib.request, "urlopen", blocked_call)
    monkeypatch.setattr(urllib.request, "build_opener", blocked_build_opener)
    monkeypatch.setattr(socket, "create_connection", blocked_call)
    for module_name in (
        "scholarly_search.crossref_client",
        "scholarly_search.source_verifier",
        "analysis.gemini_backend",
    ):
        module = import_module(module_name)
        if hasattr(module, "urlopen"):
            monkeypatch.setattr(module, "urlopen", blocked_call)
