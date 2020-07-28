import os
import pytest


@pytest.fixture(scope="session")
def url():
    return os.getenv("TENSORFLOW_URL")