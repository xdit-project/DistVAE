import pytest


@pytest.fixture
def master_port(request):
    """Unique port per test to avoid Address already in use when tests run sequentially."""
    base = 29800
    nodeid = request.node.nodeid
    return base + (hash(nodeid) % 10000)
