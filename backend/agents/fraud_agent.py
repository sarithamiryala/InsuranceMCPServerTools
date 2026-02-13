from backend.services.llm_client import llm_response
from backend.utils.safe_json import safe_json_parse
import os

def _sanitize_result(data: dict) -> dict:
    out = {}
    # fraud_score
    try:
        out["fraud_score"] = float(data.get("fraud_score", 0.0))
        if out["fraud_score"] < 0: out["fraud_score"] = 0.0
        if out["fraud_score"] > 1: out["fraud_score"] = 1.0
    except (TypeError, ValueError):
        out["fraud_score"] = 0.0

    # fraud_decision
    decision = str(data.get("fraud_decision", "SAFE")).strip().upper()
    out["fraud_decision"] = "SUSPECT" if decision == "SUSPECT" else "SAFE"
    return out

def fraud_agent(state):
    # Prepare prompt
    prompt = f"""
    You are an insurance fraud detection AI.

    Claim Amount: {state.amount}
    Claim Text: {state.extracted_text}

    Return ONLY a minified JSON object with keys:
    - "fraud_score": float between 0.0 and 1.0
    - "fraud_decision": "SAFE" or "SUSPECT"
    """

    # Fallback if API key is missing
    if not os.getenv("GOOGLE_API_KEY"):
        print("[fraud_agent] WARNING: GOOGLE_API_KEY not set. Using fallback fraud score.")
        raw_result = '{"fraud_score": 0.0, "fraud_decision": "SAFE"}'
    else:
        try:
            raw_result = llm_response(prompt)
        except Exception as e:
            print(f"[fraud_agent] ERROR calling LLM: {e}")
            raw_result = '{"fraud_score": 0.0, "fraud_decision": "SAFE"}'

    print("Raw LLM response (fraud_agent):", repr(raw_result))

    # Parse and sanitize
    fallback = {"fraud_score": 0.0, "fraud_decision": "SAFE"}
    parsed = safe_json_parse(raw_result, fallback)
    cleaned = _sanitize_result(parsed)

    # Update claim state
    state.fraud_checked = True
    state.fraud_score = cleaned["fraud_score"]
    state.fraud_decision = cleaned["fraud_decision"]

    return state
