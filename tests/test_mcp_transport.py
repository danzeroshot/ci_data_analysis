import json

from schedule_risk_agent.mcp_http_sse import handle_rpc


def test_initialize_and_tool_list():
    initialize = handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, lambda x: x)
    assert initialize["result"]["protocolVersion"] == "2024-11-05"
    tools = handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, lambda x: x)
    assert tools["result"]["tools"][0]["name"] == "score_schedule_risk"


def test_tool_call_wraps_structured_result():
    response = handle_rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "score_schedule_risk", "arguments": {"request_id": "x"}},
    }, lambda payload: {"echo": payload["request_id"]})
    assert response["result"]["structuredContent"] == {"echo": "x"}
    assert json.loads(response["result"]["content"][0]["text"]) == {"echo": "x"}

