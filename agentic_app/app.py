"""Gradio UI — Agentic Fleet Management Assistant (Milestone 2)."""

import gradio as gr
from agent import run_agent
from model_utils import SAMPLE_VEHICLES, VEHICLE_MODELS, FUEL_TYPES, TRANSMISSION_TYPES, OWNER_TYPES

RISK_COLORS = {
    "CRITICAL": "red",
    "HIGH": "orange",
    "MODERATE": "gold",
    "LOW": "green",
    "UNKNOWN": "gray",
}


def analyze_vehicle(
    api_key, sample_choice,
    vehicle_model, mileage, maintenance_history, reported_issues,
    vehicle_age, fuel_type, transmission_type, engine_size,
    odometer_reading, owner_type, insurance_premium, service_history,
    accident_history, fuel_efficiency, tire_condition, brake_condition,
    battery_status, last_service_days, warranty_expiry_days,
):
    """Run the full agent pipeline and return results."""

    vehicle_data = {
        "Vehicle_Model": vehicle_model,
        "Mileage": int(mileage),
        "Maintenance_History": maintenance_history,
        "Reported_Issues": int(reported_issues),
        "Vehicle_Age": int(vehicle_age),
        "Fuel_Type": fuel_type,
        "Transmission_Type": transmission_type,
        "Engine_Size": int(engine_size),
        "Odometer_Reading": int(odometer_reading),
        "Owner_Type": owner_type,
        "Insurance_Premium": int(insurance_premium),
        "Service_History": int(service_history),
        "Accident_History": int(accident_history),
        "Fuel_Efficiency": float(fuel_efficiency),
        "Tire_Condition": tire_condition,
        "Brake_Condition": brake_condition,
        "Battery_Status": battery_status,
        "Last_Service_Date_days": int(last_service_days),
        "Warranty_Expiry_Date_days": int(warranty_expiry_days),
    }

    result = run_agent(vehicle_data, api_key=api_key or None)

    # Build risk badge
    pred = result.get("prediction", {})
    risk = result.get("risk_level", "UNKNOWN")
    proba = pred.get("probability", 0)
    needs = pred.get("needs_maintenance", 0)
    color = RISK_COLORS.get(risk, "gray")

    risk_badge = f"""
## Risk Assessment

| Metric | Value |
|--------|-------|
| **Risk Level** | **{risk}** |
| **Maintenance Needed** | {"YES" if needs else "NO"} |
| **Probability** | {proba:.1%} |

### Top Contributing Factors
"""
    for f in pred.get("top_features", []):
        name = f["feature"].replace("num__", "").replace("ord__", "").replace("nom__", "")
        risk_badge += f"- **{name}**: importance {f['importance']:.4f}\n"

    report = result.get("report", "No report generated.")
    error = result.get("error", "")
    if error:
        report += f"\n\n> **Warning:** {error}"

    # Agent trace (shows the workflow steps)
    trace = f"""### Agent Workflow Trace
1. **analyze_vehicle** — Parsed {len(vehicle_data)} vehicle attributes
2. **predict_maintenance** — ML model prediction: risk={risk}, probability={proba:.1%}
3. **retrieve_guidelines** — Retrieved {len(result.get('guidelines', []))} relevant guidelines via RAG
4. **generate_report** — {"LLM-generated" if api_key else "Rule-based"} structured report
"""

    return risk_badge, report, trace


def load_sample(sample_name):
    """Populate form fields from a sample vehicle."""
    if sample_name not in SAMPLE_VEHICLES:
        return [gr.update()] * 19

    v = SAMPLE_VEHICLES[sample_name]
    return [
        gr.update(value=v["Vehicle_Model"]),
        gr.update(value=v["Mileage"]),
        gr.update(value=v["Maintenance_History"]),
        gr.update(value=v["Reported_Issues"]),
        gr.update(value=v["Vehicle_Age"]),
        gr.update(value=v["Fuel_Type"]),
        gr.update(value=v["Transmission_Type"]),
        gr.update(value=v["Engine_Size"]),
        gr.update(value=v["Odometer_Reading"]),
        gr.update(value=v["Owner_Type"]),
        gr.update(value=v["Insurance_Premium"]),
        gr.update(value=v["Service_History"]),
        gr.update(value=v["Accident_History"]),
        gr.update(value=v["Fuel_Efficiency"]),
        gr.update(value=v["Tire_Condition"]),
        gr.update(value=v["Brake_Condition"]),
        gr.update(value=v["Battery_Status"]),
        gr.update(value=v["Last_Service_Date_days"]),
        gr.update(value=v["Warranty_Expiry_Date_days"]),
    ]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Fleet Management AI Assistant") as demo:

    gr.Markdown("""
# Fleet Management AI Assistant
### Agentic Vehicle Maintenance Prediction & Recommendations

This system uses a **LangGraph agent workflow** that:
1. Analyzes vehicle data and validates inputs
2. Runs an **ML model** (Decision Tree + SMOTE) to predict maintenance risk
3. Retrieves relevant **maintenance guidelines** via RAG (FAISS + sentence-transformers)
4. Generates a **structured fleet management report** using an LLM (Groq/Llama)

---
""")

    with gr.Row():
        api_key = gr.Textbox(
            label="Groq API Key",
            placeholder="gsk_... (optional — rule-based report without it)",
            type="password",
            scale=3,
        )
        sample_dropdown = gr.Dropdown(
            choices=[""] + list(SAMPLE_VEHICLES.keys()),
            label="Load Sample Vehicle",
            value="",
            scale=2,
        )

    gr.Markdown("### Vehicle Details")

    with gr.Row():
        vehicle_model = gr.Dropdown(choices=VEHICLE_MODELS, label="Vehicle Model", value="Truck")
        fuel_type = gr.Dropdown(choices=FUEL_TYPES, label="Fuel Type", value="Diesel")
        transmission_type = gr.Dropdown(choices=TRANSMISSION_TYPES, label="Transmission", value="Automatic")
        owner_type = gr.Dropdown(choices=OWNER_TYPES, label="Owner Type", value="First")

    with gr.Row():
        mileage = gr.Number(label="Mileage (km)", value=60000)
        vehicle_age = gr.Number(label="Vehicle Age (years)", value=5)
        engine_size = gr.Number(label="Engine Size (cc)", value=2000)
        odometer_reading = gr.Number(label="Odometer Reading (km)", value=65000)

    with gr.Row():
        reported_issues = gr.Slider(0, 10, value=1, step=1, label="Reported Issues")
        service_history = gr.Slider(0, 20, value=6, step=1, label="Service History Count")
        accident_history = gr.Slider(0, 5, value=0, step=1, label="Accident History")
        fuel_efficiency = gr.Number(label="Fuel Efficiency (km/l)", value=14.0)

    with gr.Row():
        maintenance_history = gr.Dropdown(choices=["Poor", "Average", "Good"], label="Maintenance History", value="Average")
        tire_condition = gr.Dropdown(choices=["Worn Out", "Good", "New"], label="Tire Condition", value="Good")
        brake_condition = gr.Dropdown(choices=["Worn Out", "Good", "New"], label="Brake Condition", value="Good")
        battery_status = gr.Dropdown(choices=["Weak", "Good", "Strong"], label="Battery Status", value="Good")

    with gr.Row():
        insurance_premium = gr.Number(label="Insurance Premium", value=18000)
        last_service_days = gr.Number(label="Days Since Last Service", value=120)
        warranty_expiry_days = gr.Number(label="Days Until Warranty Expiry", value=400)

    analyze_btn = gr.Button("Run Fleet Analysis", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            risk_output = gr.Markdown(label="Risk Assessment")
            trace_output = gr.Markdown(label="Agent Trace")
        with gr.Column(scale=2):
            report_output = gr.Markdown(label="Fleet Management Report")

    # --- Wire up events ---

    form_fields = [
        vehicle_model, mileage, maintenance_history, reported_issues,
        vehicle_age, fuel_type, transmission_type, engine_size,
        odometer_reading, owner_type, insurance_premium, service_history,
        accident_history, fuel_efficiency, tire_condition, brake_condition,
        battery_status, last_service_days, warranty_expiry_days,
    ]

    sample_dropdown.change(
        fn=load_sample,
        inputs=[sample_dropdown],
        outputs=form_fields,
    )

    analyze_btn.click(
        fn=analyze_vehicle,
        inputs=[api_key, sample_dropdown] + form_fields,
        outputs=[risk_output, report_output, trace_output],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
