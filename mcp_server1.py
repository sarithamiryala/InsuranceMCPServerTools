# server/app_v4_mcp.py

from fastapi import FastAPI
from fastmcp import FastMCP
from typing import List, Optional
from datetime import datetime, timezone

# ----------------------
# Import backend modules
# ----------------------
from backend.state.claim_state import ClaimState
from backend.db.sqlite_store import (
    init_db,
    fetch_claim_and_docs,
    update_claim_fields,
    insert_documents
)
from backend.db.investigator_store import init_investigator_db
from backend.agents.registration_agent import registration_agent
from backend.graph.claim_graph_v3 import claim_graph_v3

# ----------------------
# FastAPI app init
# ----------------------
app = FastAPI(title="Enterprise Insurance Claim MCP Server")

@app.on_event("startup")
def startup():
    init_db()
    print("[DB] Initialized") 
    init_investigator_db() 
    print("[DB] investigator Initiazed")

# ----------------------
# Convert FastAPI app into MCP
# ----------------------
mcp = FastMCP.from_fastapi(app=app)

# ----------------------
# MCP Tool: Register Claim
# ----------------------
@mcp.tool
async def ClaimRegistrationTool(
    claim_id: str,
    customer_name: str,
    policy_number: str,
    description: str,
    amount: float,
    claim_type: str,
    documents: Optional[List[dict]] = None  # MCP: documents as JSON list
):
    documents = documents or []

    # Build initial state
    state = ClaimState(
        claim_id=claim_id,
        customer_name=customer_name,
        policy_number=policy_number,
        amount=amount,
        claim_type=claim_type,
        extracted_text=description,
    )

    # Run registration agent
    state = registration_agent(state)

    combined_text = description or ""

    # Process MCP JSON documents
    for doc in documents:
        state.documents.append(doc)
        combined_text += f"\n\n[DOCUMENT: {doc['doc_type'].upper()}]\n{doc['extracted_text']}"

        # Persist in DB
        insert_documents(
            transaction_id=state.transaction_id,
            filename=doc["filename"],
            content_type=doc.get("content_type"),
            size_bytes=doc.get("size_bytes", 0),
            doc_type=doc.get("doc_type"),
            extracted_text=doc.get("extracted_text")
        )

    # Store aggregated extracted text
    update_claim_fields(
        state.transaction_id,
        extracted_text=combined_text,
        status="REGISTERED",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    state.extracted_text = combined_text

    return {
        "transaction_id": state.transaction_id,
        "registered_at": state.registered_at,
        "documents_uploaded": len(documents),
        "claim_id": state.claim_id,
        "policy_number": state.policy_number
    }

# ----------------------
# MCP Tool: Check Status
# ----------------------
@mcp.tool
def ClaimStatusTool(transaction_id: str):
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Transaction not found"}

    return {
        "transaction_id": claim["transaction_id"],
        "claim_id": claim["claim_id"],
        "policy_number": claim["policy_number"],
        "status": claim["status"],
        "final_decision": claim.get("final_decision"),
        "documents_uploaded": len(docs) if docs else 0
    }

# ----------------------
# MCP Tool: AI Processing
# ----------------------
@mcp.tool
async def ManagerProcessingTool(transaction_id: str):
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    state = ClaimState(
        transaction_id=claim["transaction_id"],
        claim_id=claim["claim_id"],
        customer_name=claim["customer_name"],
        policy_number=claim["policy_number"],
        amount=claim["amount"],
        claim_type=claim["claim_type"],
        extracted_text=claim["extracted_text"],
        claim_registered=True,
        registered_at=claim["registered_at"]
    )

    # Run AI workflow
    final_state = await claim_graph_v3.ainvoke(state)
    if not isinstance(final_state, dict):
        final_state = final_state.model_dump()

    # Determine risk level
    fraud_score = final_state.get("fraud_score", 0)
    if fraud_score < 0.3:
        risk_level = "LOW"
    elif fraud_score < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # Persist AI results
    update_claim_fields(
        transaction_id,
        final_decision=final_state.get("final_decision"),
        fraud_score=fraud_score,
        fraud_decision=final_state.get("fraud_decision"),
        ai_explanation=final_state.get("ai_explanation"),
        status=final_state.get("final_decision") or "UNDER_REVIEW",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": transaction_id,
        "final_decision": final_state.get("final_decision"),
        "fraud_score": fraud_score,
        "fraud_decision": final_state.get("fraud_decision"),
        "validation": final_state.get("validation"),
        "assignment": final_state.get("assignment"),
        "ai_explanation": final_state.get("ai_explanation"),
        "risk_level": risk_level
    }

# ----------------------
# MCP Tool: Manager Decision (Human Override)
# ----------------------
@mcp.tool
def ManagerDecisionTool(transaction_id: str, decision: str, comment: Optional[str] = None):
    claim, _ = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    decision = decision.upper()
    valid_decisions = ["APPROVED", "REJECTED", "PENDING_DOCUMENTS"]
    if decision not in valid_decisions:
        return {"error": "Invalid decision"}

    # Append audit log
    audit_msg = f"Manager decision: {decision}"
    if comment:
        audit_msg += f" | Comment: {comment}"
    update_claim_fields(
        transaction_id,
        final_decision=decision,
        status=decision,
        manager_comment=comment,
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": transaction_id,
        "status": decision,
        "message": f"Claim has been {decision.replace('_', ' ').lower()} successfully.",
        "audit_log": audit_msg
    }

# ----------------------
# Run MCP server
# ----------------------
if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000)
