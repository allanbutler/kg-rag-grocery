from fastapi.testclient import TestClient
from sgs.app import app

def test_health():
    c = TestClient(app)
    assert c.get("/health").json()["ok"] is True
