import os
import pytest


@pytest.fixture(scope="session")
def url():
    fetch_url = os.getenv("TENSORFLOW_URL")
    assert fetch_url is not None, "not define env \"TENSORFLOW_URL\""
    return os.getenv("TENSORFLOW_URL")
