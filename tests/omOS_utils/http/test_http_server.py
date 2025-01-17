import pytest
import json
import requests
import threading
import time
from unittest.mock import Mock
from typing import Generator

from omOS_utils import http

def wait_for_server(url: str, timeout: int = 5, interval: float = 0.1) -> bool:
    """Wait for server to be ready to accept connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            requests.post(url, json={"ping": "test"}, timeout=0.5)
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(interval)
    return False

@pytest.fixture
def server() -> Generator[http.Server, None, None]:
    """Fixture that creates and manages a test server instance."""
    test_server = http.Server(port=6792)  # Use different port for testing
    test_server.start()

    # Wait for server to be ready
    server_url = "http://localhost:6792"
    if not wait_for_server(server_url):
        pytest.fail("Server failed to start within timeout period")

    yield test_server

    test_server.stop()
    # Wait for server to fully shut down
    time.sleep(0.5)

@pytest.fixture
def server_url() -> str:
    """Fixture that returns the test server URL."""
    return "http://localhost:6792"

def test_server_initialization():
    """Test server initialization with custom parameters."""
    server = http.Server(host="127.0.0.1", port=8000, timeout=30)
    assert server.host == "127.0.0.1"
    assert server.port == 8000
    assert server.timeout == 30
    assert server.running is True
    assert server.server is None
    assert server.message_callback is None

def test_callback_registration(server: http.Server):
    """Test registering a message callback."""
    mock_callback = Mock(return_value={"status": "ok"})
    server.register_message_callback(mock_callback)
    assert server.message_callback == mock_callback

def test_post_request_with_callback(server: http.Server, server_url: str):
    """Test handling a POST request with a registered callback."""
    # Register mock callback
    mock_response = {"status": "success", "data": "test"}
    mock_callback = Mock(return_value=mock_response)
    server.register_message_callback(mock_callback)

    # Send test request
    test_data = {"message": "test"}
    response = requests.post(
        f"{server_url}/test",
        json=test_data,
        timeout=5
    )

    assert response.status_code == 200
    assert response.json() == mock_response
    mock_callback.assert_called_once_with(test_data, "/test")

def test_get_request_not_allowed(server: http.Server, server_url: str):
    """Test that GET requests are not allowed."""
    response = requests.get(server_url, timeout=5)
    assert response.status_code == 405
    assert "error" in response.json()

def test_post_request_without_callback(server: http.Server, server_url: str):
    """Test handling a POST request without a registered callback."""
    response = requests.post(
        server_url,
        json={"test": "data"},
        timeout=5
    )
    assert response.status_code == 500
    assert response.json()["error"] == "No callback handler registered"

def test_post_request_invalid_json(server: http.Server, server_url: str):
    """Test handling a POST request with invalid JSON."""
    response = requests.post(
        server_url,
        data="invalid json",
        headers={"Content-Type": "application/json"},
        timeout=5
    )
    assert response.status_code == 400
    assert "error" in response.json()

def test_callback_string_response(server: http.Server, server_url: str):
    """Test handling a string response from the callback."""
    mock_callback = Mock(return_value="test response")
    server.register_message_callback(mock_callback)

    response = requests.post(
        server_url,
        json={"test": "data"},
        timeout=5
    )

    assert response.status_code == 200
    assert response.json() == {"response": "test response"}

def test_server_stop():
    """Test server stop functionality."""
    server = http.Server(port=6793)
    server.start()
    time.sleep(0.1)

    server.stop()
    assert server.running is False

    # Verify server is no longer accepting connections
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.post(
            "http://localhost:6793",
            json={"test": "data"},
            timeout=1
        )

@pytest.mark.parametrize("test_input,expected_code", [
    ({"valid": "data"}, 200),
    (b"invalid data", 400),
    (None, 400),
])
def test_various_inputs(server: http.Server, server_url: str, test_input: http.JsonDict, expected_code: int):
    """Test handling various types of input data."""
    mock_callback = Mock(return_value={"status": "ok"})
    server.register_message_callback(mock_callback)

    if isinstance(test_input, dict):
        response = requests.post(server_url, json=test_input, timeout=5)
    else:
        response = requests.post(server_url, data=test_input, timeout=5)

    assert response.status_code == expected_code