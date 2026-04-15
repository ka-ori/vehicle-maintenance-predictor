"""Gradio UI — Agentic Fleet Management Assistant (Milestone 2)."""

import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from agent import run_agent
from model_utils import SAMPLE_VEHICLES, VEHICLE_MODELS, FUEL_TYPES, TRANSMISSION_TYPES, OWNER_TYPES


# ── CSS (matching mid-sem style) ────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

* { font-family: 'JetBrains Mono', monospace !important; }

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

#header {
    border-bottom: 1px solid #333;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

#header h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 0 0.25rem 0 !important;
}

#header p { font-size: 0.8rem !important; opacity: 0.55; margin: 0 !important; }

.section-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    opacity: 0.45;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.4rem;
}

#analyze-btn {
    margin-top: 1.5rem;
    border-radius: 2px !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

label span { font-size: 0.7rem !important; letter-spacing: 0.05em !important; opacity: 0.7; }

.report-output { min-height: 400px; }
"""


# ── Prediction handler ──────────────────────────────────────────
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

    pred = result.get("prediction", {})
    risk = result.get("risk_level", "UNKNOWN")
    proba = pred.get("probability", 0)
    needs = pred.get("needs_maintenance", 0)

    risk_badge = f"""## Risk Assessment

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


# ── Gradio UI ───────────────────────────────────────────────────
def build_app():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Fleet Management AI Assistant",
        theme=gr.themes.Monochrome(),
        css=CSS,
    ) as demo:

        with gr.Column(elem_id="header"):
            gr.Markdown("# Fleet Management AI Assistant")
            gr.Markdown("Agentic vehicle maintenance prediction & recommendations")

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

        with gr.Row():
            with gr.Column():
                gr.Markdown("Vehicle Profile", elem_classes="section-label")
                vehicle_model = gr.Dropdown(choices=VEHICLE_MODELS, label="Vehicle Model", value="Truck")
                fuel_type = gr.Dropdown(choices=FUEL_TYPES, label="Fuel Type", value="Diesel")
                transmission_type = gr.Dropdown(choices=TRANSMISSION_TYPES, label="Transmission", value="Automatic")
                owner_type = gr.Dropdown(choices=OWNER_TYPES, label="Owner Type", value="First")
                engine_size = gr.Dropdown(choices=[1000, 1500, 2000, 2500, 3000, 3500], label="Engine Size (cc)", value=2000)

                gr.Markdown("Condition", elem_classes="section-label")
                maintenance_history = gr.Dropdown(choices=["Poor", "Average", "Good"], label="Maintenance History", value="Average")
                tire_condition = gr.Dropdown(choices=["Worn Out", "Good", "New"], label="Tire Condition", value="Good")
                brake_condition = gr.Dropdown(choices=["Worn Out", "Good", "New"], label="Brake Condition", value="Good")
                battery_status = gr.Dropdown(choices=["Weak", "Good", "Strong"], label="Battery Status", value="Good")

            with gr.Column():
                gr.Markdown("Usage & History", elem_classes="section-label")
                mileage = gr.Slider(5000, 150000, step=1000, value=60000, label="Mileage (km)")
                vehicle_age = gr.Slider(1, 20, step=1, value=5, label="Vehicle Age (years)")
                odometer_reading = gr.Slider(5000, 250000, step=1000, value=65000, label="Odometer Reading (km)")
                fuel_efficiency = gr.Slider(8.0, 22.0, step=0.1, value=14.0, label="Fuel Efficiency (km/l)")
                reported_issues = gr.Slider(0, 10, step=1, value=1, label="Reported Issues")
                service_history = gr.Slider(0, 20, step=1, value=6, label="Service History (count)")
                accident_history = gr.Slider(0, 5, step=1, value=0, label="Accident History (count)")

                gr.Markdown("Financial & Service", elem_classes="section-label")
                insurance_premium = gr.Slider(8000, 35000, step=500, value=18000, label="Insurance Premium")
                last_service_days = gr.Slider(30, 900, step=10, value=120, label="Days Since Last Service")
                warranty_expiry_days = gr.Slider(-400, 800, step=10, value=400, label="Days Until Warranty Expiry")

        analyze_btn = gr.Button("Run Fleet Analysis", variant="primary", size="lg", elem_id="analyze-btn")

        gr.Markdown("Result", elem_classes="section-label")

        with gr.Row():
            with gr.Column(scale=1):
                risk_output = gr.Markdown(label="Risk Assessment")
                trace_output = gr.Markdown(label="Agent Trace")
            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Fleet Management Report", elem_classes="report-output")

        # ── Wire up events ──
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

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
