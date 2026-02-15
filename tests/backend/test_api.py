"""Tests for FastAPI API endpoints."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.api.routes import router, set_data_loader, set_graph, set_session_store
from backend.api.session import SessionStore
from backend.data.loader import ArgoDataLoader


@pytest.fixture
def session_store():
    store = SessionStore()
    set_session_store(store)
    return store


@pytest.fixture
def mock_data_loader():
    loader = ArgoDataLoader.__new__(ArgoDataLoader)
    loader._settings = None
    set_data_loader(loader)
    return loader


@pytest.fixture
def mock_graph():
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "messages": [AIMessage(content="The Argo program is a global array of floats.")],
        "intent": "info",
        "data": {"sources": ["argo_overview"]},
        "visualization": {},
        "metadata": {"agent_path": ["supervisor", "rag_agent"]},
    }
    set_graph(graph)
    return graph


@pytest.fixture
def client(session_store, mock_data_loader, mock_graph):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestChatMessage:
    def test_post_chat_message(self, client, mock_graph):
        response = client.post("/api/v1/chat/message", json={
            "session_id": "test-session",
            "message": "What is Argo?",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert "content" in data
        assert len(data["content"]) > 0
        assert data["agent_path"] == ["supervisor", "rag_agent"]

    def test_empty_message_rejected(self, client):
        response = client.post("/api/v1/chat/message", json={
            "session_id": "test-session",
            "message": "   ",
        })
        assert response.status_code == 400

    def test_creates_session(self, client, session_store):
        sid = str(uuid.uuid4())
        response = client.post("/api/v1/chat/message", json={
            "session_id": sid,
            "message": "Hello",
        })
        assert response.status_code == 200
        assert session_store.session_exists(sid)

    def test_stores_messages(self, client, session_store):
        sid = "store-test"
        client.post("/api/v1/chat/message", json={
            "session_id": sid,
            "message": "What is Argo?",
        })
        history = session_store.get_history(sid)
        assert len(history) == 2  # user + assistant
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_visualization_in_response(self, client, mock_graph):
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Here is a chart.")],
            "intent": "viz",
            "data": {},
            "visualization": {
                "chart_type": "bar_chart",
                "plotly_json": {"data": [], "layout": {}},
                "description": "Test chart",
            },
            "metadata": {"agent_path": ["supervisor", "query_agent", "viz_agent"]},
        }
        response = client.post("/api/v1/chat/message", json={
            "session_id": "viz-test",
            "message": "Plot temperature",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["visualization"] is not None
        assert data["visualization"]["chart_type"] == "bar_chart"

    def test_no_graph_returns_503(self, client):
        set_graph(None)
        response = client.post("/api/v1/chat/message", json={
            "session_id": "test",
            "message": "Hello",
        })
        assert response.status_code == 503


class TestChatHistory:
    def test_get_history(self, client, session_store):
        sid = "history-test"
        session_store.get_or_create_session(sid)
        session_store.add_message(sid, "user", "Hello")
        session_store.add_message(sid, "assistant", "Hi there!")

        response = client.get(f"/api/v1/chat/history/{sid}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"

    def test_unknown_session_404(self, client):
        response = client.get("/api/v1/chat/history/nonexistent")
        assert response.status_code == 404


class TestDataEndpoints:
    def test_list_variables(self, client):
        response = client.get("/api/v1/data/variables")
        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert len(data["variables"]) == 4
        names = {v["name"] for v in data["variables"]}
        assert "TEMP" in names
        assert "PSAL" in names

    def test_variable_has_fields(self, client):
        response = client.get("/api/v1/data/variables")
        var = response.json()["variables"][0]
        assert "name" in var
        assert "display_name" in var
        assert "unit" in var
        assert "description" in var
        assert "typical_range" in var

    def test_metadata(self, client):
        response = client.get("/api/v1/data/metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["lat_bounds"] == [-90.0, 90.0]
        assert data["lon_bounds"] == [-180.0, 180.0]
        assert "TEMP" in data["available_variables"]
        assert data["data_source"] == "Argo GDAC"

    def test_no_loader_returns_503(self, client):
        set_data_loader(None)
        response = client.get("/api/v1/data/variables")
        assert response.status_code == 503


class TestSessionStore:
    def test_create_session(self):
        store = SessionStore()
        sid = store.get_or_create_session()
        assert store.session_exists(sid)

    def test_get_existing_session(self):
        store = SessionStore()
        sid = store.get_or_create_session("my-session")
        sid2 = store.get_or_create_session("my-session")
        assert sid == sid2

    def test_add_and_get_messages(self):
        store = SessionStore()
        sid = store.get_or_create_session("test")
        store.add_message(sid, "user", "Hello")
        store.add_message(sid, "assistant", "Hi!")
        history = store.get_history(sid)
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi!"

    def test_message_has_timestamp(self):
        store = SessionStore()
        sid = store.get_or_create_session("test")
        msg = store.add_message(sid, "user", "Hello")
        assert msg.timestamp != ""

    def test_empty_session_history(self):
        store = SessionStore()
        history = store.get_history("nonexistent")
        assert history == []
