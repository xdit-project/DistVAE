import zlib

import pytest


@pytest.fixture
def master_port(request):
    """A port for this test's gloo rendezvous, distinct from every other test's

    Below the ephemeral range Linux hands out to outgoing connections, so a port picked here is
    not one the kernel might have just given to something else. Keyed on the test's id by a hash
    that does not change between runs, so a failure is reproducible; run_distributed retries on a
    fresh port anyway, for the collision this cannot rule out.
    """
    base = 20000
    return base + zlib.crc32(request.node.nodeid.encode()) % 10000
