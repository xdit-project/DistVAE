import zlib

import pytest


@pytest.fixture
def master_port(request):
    """Return a deterministic port for this test's Gloo rendezvous.

    Ports 20000–29999 are below Linux's default ephemeral range. CRC32 makes the initial port
    stable for each test across runs. `run_distributed` retries if two tests still collide.
    """
    base = 20000
    return base + zlib.crc32(request.node.nodeid.encode()) % 10000
