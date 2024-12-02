import uuid
from typing import Any, Dict

from simulation.nets.messages import RequestMessage, ResponseMessage, Status, generate_request_id


class TestMessages:
    def test_generate_request_id(self):
        """Test request ID generation."""
        request_id_1 = generate_request_id()
        request_id_2 = generate_request_id()

        # Validate UUID format
        assert isinstance(request_id_1, str)
        assert len(request_id_1) == 36  # Standard UUID length

        # Ensure uniqueness
        assert request_id_1 != request_id_2

    def test_status_enum(self):
        """Test Status enum values."""
        assert Status.SUCCESS.value == "success"
        assert Status.ERROR.value == "error"
        assert Status.WARNING.value == "warning"

    def test_request_message_creation(self):
        """Test RequestMessage creation."""
        request_id = generate_request_id()
        payload: Dict[str, Any] = {"param1": 42, "param2": "test"}

        request = RequestMessage(request_id=request_id, command="start", payload=payload)

        assert request.request_id == request_id
        assert request.command == "start"
        assert request.payload == payload

    def test_request_message_default_values(self):
        """Test RequestMessage with default values."""
        request_id = generate_request_id()

        request = RequestMessage(request_id=request_id, command="stop")

        assert request.request_id == request_id
        assert request.command == "stop"
        assert request.payload is None

    def test_response_message_creation(self):
        """Test ResponseMessage creation."""
        request_id = generate_request_id()
        result: Dict[str, Any] = {"simulation_id": "sim123"}

        response = ResponseMessage(
            request_id=request_id,
            status=Status.SUCCESS.value,
            message="Simulation started",
            result=result,
        )

        assert response.request_id == request_id
        assert response.status == Status.SUCCESS.value
        assert response.message == "Simulation started"
        assert response.result == result

    def test_response_message_default_values(self):
        """Test ResponseMessage with default values."""
        request_id = generate_request_id()

        response = ResponseMessage(request_id=request_id, status=Status.ERROR.value)

        assert response.request_id == request_id
        assert response.status == Status.ERROR.value
        assert response.message is None
        assert response.result is None

    def test_response_message_ensure_request_id(self):
        """Test ensure_request_id method."""
        request_id = generate_request_id()

        response = ResponseMessage(request_id="", status=Status.SUCCESS.value)

        response.ensure_request_id(request_id)

        assert response.request_id == request_id

    def test_response_message_ensure_result_id(self):
        """Test ensure_result_id method."""
        response = ResponseMessage(
            request_id=generate_request_id(), status=Status.SUCCESS.value, result={}
        )

        response.ensure_result_id()

        assert "id" in response.result
        assert isinstance(response.result["id"], str)
        assert len(response.result["id"]) == 36  # Standard UUID length

    def test_response_message_ensure_result_id_existing(self):
        """Test ensure_result_id method with existing ID."""
        existing_id = str(uuid.uuid4())
        response = ResponseMessage(
            request_id=generate_request_id(),
            status=Status.SUCCESS.value,
            result={"id": existing_id},
        )

        response.ensure_result_id()

        assert response.result["id"] == existing_id
