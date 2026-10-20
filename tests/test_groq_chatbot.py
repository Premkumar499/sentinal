import pytest
import json
from src.chatbot_engine import InfoChatbotEngine, normalize_confidence, groq_llm
from app.app import app

def test_normalize_confidence():
    # Should clamp properly and never exceed 100%
    assert 5.0 <= normalize_confidence(2435.08, 31444480.76) <= 99.5
    assert normalize_confidence(0.85, 5000.0) == 85.0
    assert 5.0 <= normalize_confidence(0.0, 0.0) <= 99.5
    assert normalize_confidence(1.5, 200.0) <= 99.5

def test_parse_query_regex_fallback():
    parsed = InfoChatbotEngine.parse_query_regex("tell me the area which is new building is come in the VEGETATION area")
    assert parsed["t1_class"] == "vegetation"
    assert parsed["t2_class"] == "building"

def test_parse_query_with_district():
    parsed = InfoChatbotEngine.parse_query("the karur vegetation converted to building area and statistics")
    assert parsed["district"] == "karur"
    assert parsed["t2_class"] == "building"

def test_answer_query_grounding():
    resp = InfoChatbotEngine.answer_query("the karur vegetation converted to building area and statistics")
    assert resp["status"] == "success"
    assert "metrics" in resp
    assert resp["metrics"]["scope"] == "Karur"
    assert resp["metrics"]["total_parcels"] > 0
    assert resp["metrics"]["total_hectares"] > 0
    # Confidence must not exceed 100%
    conf_str = resp["metrics"]["average_confidence"]
    conf_val = float(conf_str.replace("%", ""))
    assert 0.0 <= conf_val <= 100.0

def test_flask_chatbot_api():
    client = app.test_client()
    res = client.post("/api/chatbot/info", json={
        "query": "the karur vegetation converted to building area and statistics",
        "history": []
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "metrics" in data

def test_conversational_greeting():
    resp = InfoChatbotEngine.answer_query("hi")
    assert resp["status"] == "success"
    assert resp.get("is_conversational") is True
    assert "metrics" in resp
    # Should NOT dump statewide 440k parcels for a simple 'hi'
    assert resp["metrics"]["total_parcels"] == 0
    assert len(resp["ai_briefing"]) > 10
    assert "suggested_prompts" in resp

