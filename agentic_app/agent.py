"""LangGraph agentic workflow for fleet management analysis."""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from model_utils import predict
from rag_utils import get_rag


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class FleetState(TypedDict):
    vehicle_data: dict
    vehicle_summary: str
    prediction: dict
    risk_level: str
    guidelines: list[str]
    report: str
    error: str


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_llm(api_key: str | None = None):
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        temperature=0.3,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def analyze_vehicle(state: FleetState) -> dict:
    """Node 1: Parse and summarize vehicle data."""
    v = state["vehicle_data"]
    lines = []
    for key, val in v.items():
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {val}")
    summary = "\n".join(lines)
    return {"vehicle_summary": summary}


def predict_maintenance(state: FleetState) -> dict:
    """Node 2: Run ML model prediction."""
    try:
        result = predict(state["vehicle_data"])
        return {
            "prediction": result,
            "risk_level": result["risk_level"],
        }
    except Exception as e:
        return {
            "prediction": {},
            "risk_level": "UNKNOWN",
            "error": f"Prediction error: {e}",
        }


def retrieve_guidelines(state: FleetState) -> dict:
    """Node 3: RAG retrieval of relevant maintenance guidelines."""
    v = state["vehicle_data"]
    risk = state.get("risk_level", "UNKNOWN")

    # Build a retrieval query from vehicle condition
    query_parts = [f"vehicle maintenance for {v.get('Vehicle_Model', 'vehicle')}"]
    query_parts.append(f"risk level {risk}")

    tire = v.get("Tire_Condition", "")
    brake = v.get("Brake_Condition", "")
    battery = v.get("Battery_Status", "")

    if tire == "Worn Out":
        query_parts.append("tire replacement worn out")
    if brake == "Worn Out":
        query_parts.append("brake maintenance worn brakes")
    if battery == "Weak":
        query_parts.append("battery replacement weak battery")
    if v.get("Mileage", 0) > 100000:
        query_parts.append("high mileage vehicle maintenance")
    if v.get("Last_Service_Date_days", 0) > 300:
        query_parts.append("overdue service maintenance scheduling")

    query = ". ".join(query_parts)

    try:
        rag = get_rag()
        docs = rag.retrieve(query, k=4)
        return {"guidelines": docs}
    except Exception as e:
        return {"guidelines": [], "error": f"RAG error: {e}"}


def generate_report(state: FleetState) -> dict:
    """Node 4: LLM generates structured fleet management report."""
    api_key = state["vehicle_data"].get("_api_key")
    llm = _get_llm(api_key)

    pred = state.get("prediction", {})
    risk = state.get("risk_level", "UNKNOWN")
    guidelines = state.get("guidelines", [])
    vehicle_summary = state.get("vehicle_summary", "N/A")

    guidelines_text = "\n\n---\n\n".join(guidelines) if guidelines else "No guidelines retrieved."

    if llm is None:
        # Fallback: generate report without LLM
        report = _generate_fallback_report(state)
        return {"report": report}

    prompt = f"""You are an AI Fleet Management Assistant. Based on the vehicle data,
ML prediction results, and retrieved maintenance guidelines, generate a structured
fleet management report.

## Vehicle Data
{vehicle_summary}

## ML Prediction Results
- Needs Maintenance: {"YES" if pred.get("needs_maintenance") else "NO"}
- Maintenance Probability: {pred.get("probability", "N/A")}
- Risk Level: {risk}
- Top Contributing Factors: {pred.get("top_features", [])}

## Retrieved Maintenance Guidelines
{guidelines_text}

---

Generate a report with EXACTLY these sections:

### HEALTH SUMMARY
Provide a clear assessment of the vehicle's current health status and risk level.
Explain what the ML model found and why.

### ACTION PLAN
List specific maintenance actions needed, ordered by priority (critical first).
Include estimated timelines for each action.

### MAINTENANCE SCHEDULE
Provide a recommended maintenance schedule going forward (immediate, 30 days,
90 days, 6 months).

### COST ESTIMATE
Provide rough cost ranges for recommended maintenance actions.

### SOURCES
List which maintenance guidelines informed this report.

### DISCLAIMER
Include an operational safety disclaimer that this is an AI-assisted recommendation
and should be verified by a certified mechanic before acting on critical maintenance.

Keep the report professional, actionable, and concise."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a professional fleet management AI assistant. "
                          "Provide accurate, actionable maintenance recommendations. "
                          "Always prioritize safety. Be specific with timelines and actions."),
            HumanMessage(content=prompt),
        ])
        return {"report": response.content}
    except Exception as e:
        report = _generate_fallback_report(state)
        report += f"\n\n> *Note: LLM generation failed ({e}). Showing rule-based report.*"
        return {"report": report}


def _generate_fallback_report(state: FleetState) -> str:
    """Rule-based report when LLM is unavailable."""
    pred = state.get("prediction", {})
    risk = state.get("risk_level", "UNKNOWN")
    v = state.get("vehicle_data", {})
    guidelines = state.get("guidelines", [])

    needs = pred.get("needs_maintenance", 0)
    proba = pred.get("probability", 0)

    actions = []
    if v.get("Tire_Condition") == "Worn Out":
        actions.append("**[CRITICAL]** Replace worn-out tires immediately")
    if v.get("Brake_Condition") == "Worn Out":
        actions.append("**[CRITICAL]** Inspect and replace brake pads/rotors")
    if v.get("Battery_Status") == "Weak":
        actions.append("**[HIGH]** Test battery voltage and replace if below 12.4V")
    if v.get("Maintenance_History") == "Poor":
        actions.append("**[HIGH]** Schedule comprehensive service — poor maintenance history")
    if v.get("Last_Service_Date_days", 0) > 300:
        actions.append("**[HIGH]** Overdue for service — last service was over 300 days ago")
    if v.get("Mileage", 0) > 100000:
        actions.append("**[MODERATE]** High-mileage inspection recommended")
    if v.get("Accident_History", 0) > 1:
        actions.append("**[MODERATE]** Structural inspection due to accident history")

    if not actions:
        actions.append("No critical actions needed — continue regular maintenance schedule")

    report = f"""### HEALTH SUMMARY
- **Risk Level:** {risk}
- **Maintenance Needed:** {"YES" if needs else "NO"}
- **Probability:** {proba:.1%}
- **Vehicle:** {v.get("Vehicle_Model", "N/A")} | Age: {v.get("Vehicle_Age", "N/A")} years | Mileage: {v.get("Mileage", "N/A"):,} km

### ACTION PLAN
{chr(10).join(f"{i+1}. {a}" for i, a in enumerate(actions))}

### MAINTENANCE SCHEDULE
- **Immediate:** Address all CRITICAL items before next trip
- **30 days:** Complete HIGH priority items
- **90 days:** Address MODERATE items and schedule routine inspection
- **6 months:** Full comprehensive service

### SOURCES
Retrieved {len(guidelines)} relevant maintenance guidelines from the knowledge base.

### DISCLAIMER
This is an AI-generated maintenance recommendation based on predictive analytics.
All critical maintenance decisions should be verified by a certified automotive
technician. Do not rely solely on this report for safety-critical decisions.
"""
    return report


# ---------------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------------

def route_by_risk(state: FleetState) -> str:
    """Route to different paths based on risk level."""
    risk = state.get("risk_level", "UNKNOWN")
    if risk in ("CRITICAL", "HIGH"):
        return "retrieve_guidelines"
    return "retrieve_guidelines"  # Always retrieve, but could add an emergency node


# ---------------------------------------------------------------------------
# Build the agent graph
# ---------------------------------------------------------------------------

def build_agent():
    """Compile the LangGraph workflow."""
    workflow = StateGraph(FleetState)

    workflow.add_node("analyze_vehicle", analyze_vehicle)
    workflow.add_node("predict_maintenance", predict_maintenance)
    workflow.add_node("retrieve_guidelines", retrieve_guidelines)
    workflow.add_node("generate_report", generate_report)

    workflow.set_entry_point("analyze_vehicle")
    workflow.add_edge("analyze_vehicle", "predict_maintenance")
    workflow.add_conditional_edges(
        "predict_maintenance",
        route_by_risk,
        {"retrieve_guidelines": "retrieve_guidelines"},
    )
    workflow.add_edge("retrieve_guidelines", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_agent(vehicle_data: dict, api_key: str | None = None) -> dict:
    """Execute the full agent workflow for a vehicle.

    Returns the final state dict with all intermediate results.
    """
    # Pass API key through vehicle_data (cleaned before ML prediction)
    data = {**vehicle_data, "_api_key": api_key}

    agent = get_agent()
    result = agent.invoke({
        "vehicle_data": data,
        "vehicle_summary": "",
        "prediction": {},
        "risk_level": "",
        "guidelines": [],
        "report": "",
        "error": "",
    })
    return result
