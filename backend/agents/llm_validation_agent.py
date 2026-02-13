import json
from backend.state.claim_state import ClaimState, ValidationResult
from backend.services.llm_client import llm_response

# ============================================================
# FALLBACK RULE-BASED VALIDATION
# ============================================================

def _fallback_validation(state: ClaimState) -> ClaimState:
    required = {
        "motor": ["fir", "itemized_invoice", "payment_receipt", "id_proof"],
        "health": ["discharge_summary", "itemized_invoice", "payment_receipt", "id_proof"]
    }.get((state.claim_type or "").lower(), ["itemized_invoice", "payment_receipt", "id_proof"])

    present = {d.doc_type for d in state.documents if d.doc_type}
    missing = [r for r in required if r not in present]

    vr = ValidationResult(
        required_missing=missing,
        warnings=[],
        errors=[],
        docs_ok=(len(missing) == 0),
    )

    state.validation = vr
    state.claim_validated = vr.docs_ok
    state.logs.append("[validation_llm] Fallback rule-based used.")
    return state


# ============================================================
# LLM VALIDATION AGENT WITH ENHANCED LOGGING
# ============================================================

def llm_validation_agent(state: ClaimState) -> ClaimState:

    state.logs.append("[validation_llm] start")

    prompt = f"""
You are an expert insurance claim validator.

Return STRICT minified JSON only (no markdown).

{{
  "documents_detected": {{"itemized_invoice": false,"payment_receipt": false,"fir": false,"id_proof": false,"discharge_summary": false}},
  "missing_documents": [],
  "fields_extracted": {{"invoice_number": null,"invoice_total": null,"invoice_date": null}},
  "amount_matches_claim": false,
  "validation_passed": false,
  "warnings": [],
  "errors": []
}}

### CLAIM DETAILS ###
claim_type = "{state.claim_type}"
claim_amount = "{state.amount}"

### OCR DOCUMENTS ###
"""

    for i, doc in enumerate(state.documents):
        prompt += f"\n\n### DOC_{i+1} ({doc.filename}, {doc.doc_type}) ###\n{doc.extracted_text or ''}\n"

    try:
        raw = llm_response(prompt)

        # --------------------------------------------------
        # 1️⃣ Empty response
        # --------------------------------------------------
        if not raw:
            state.logs.append("[validation_llm] empty response -> fallback")
            return _fallback_validation(state)

        # --------------------------------------------------
        # 2️⃣ Dict response (API error)
        # --------------------------------------------------
        if isinstance(raw, dict):
            state.logs.append(f"[validation_llm] dict response detected -> fallback: {raw}")
            return _fallback_validation(state)

        # --------------------------------------------------
        # 3️⃣ Non-string response
        # --------------------------------------------------
        if not isinstance(raw, str):
            state.logs.append(f"[validation_llm] invalid response type -> fallback: {type(raw)}")
            return _fallback_validation(state)

        lower_raw = raw.lower()

        # --------------------------------------------------
        # 4️⃣ Rate limit / quota
        # --------------------------------------------------
        if any(x in lower_raw for x in ["resource_exhausted", "quota", "rate limit", "429"]):
            state.logs.append(f"[validation_llm] rate limit detected -> fallback: {raw}")
            return _fallback_validation(state)

        # --------------------------------------------------
        # 5️⃣ Parse JSON safely
        # --------------------------------------------------
        try:
            parsed = json.loads(raw)
        except Exception as e:
            state.logs.append(f"[validation_llm] JSON parse failed -> fallback: {e} | raw={raw}")
            return _fallback_validation(state)

        # --------------------------------------------------
        # 6️⃣ Extract fields safely
        # --------------------------------------------------
        missing = parsed.get("missing_documents", []) or []
        errors = parsed.get("errors", []) or []
        warnings = parsed.get("warnings", []) or []
        passed = bool(parsed.get("validation_passed", False))

        vr = ValidationResult(
            required_missing=missing,
            errors=errors,
            warnings=warnings,
            docs_ok=passed
        )

        state.validation = vr
        state.claim_validated = passed
        extracted = parsed.get("fields_extracted", {})

        state.logs.append(f"[validation_llm] extracted={extracted}")
        state.logs.append(f"[validation_llm] missing={missing}")
        state.logs.append(f"[validation_llm] errors={errors}")

        return state

    except Exception as e:
        # Capture exception type, message, and optionally raw LLM output
        state.logs.append(f"[validation_llm] exception={type(e).__name__}: {e} -> fallback")
        if 'raw' in locals():
            state.logs.append(f"[validation_llm] last LLM output (partial)={raw}")
        return _fallback_validation(state)
