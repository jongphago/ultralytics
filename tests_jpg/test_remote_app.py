import time
import httpx
import pytest

external_ip = "34.47.64.171"
REMOTE_SERVER_URL = f"http://{external_ip}:8000/is_cuda_available"


@pytest.mark.asyncio
async def test_remote_server_response_time():
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        response = await client.get(REMOTE_SERVER_URL)

    end_time = time.perf_counter()
    duration = end_time - start_time

    assert response.status_code == 200
    print(f"Response: {response.json()}, Duration: {duration:.4f} seconds")

if __name__ == "__main__":
    pytest.main(["-s", "-vv", "tests_jpg/test_remote_app.py"])
