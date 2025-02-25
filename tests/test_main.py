from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_mock_image_generation():
    response = client.post("/generate-image-test/", json={"text": "a magical forest"})
    assert response.status_code == 200
    assert "image_url" in response.json()
    assert response.json()["image_url"].startswith("https://placehold.co/")
