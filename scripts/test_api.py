"""Quick test that API loads and health endpoint works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)
r = client.get("/api/health")
print("Health status:", r.status_code)
print("Response:", r.json())
