"""RAG pipeline — FAISS vector store with vehicle maintenance knowledge base."""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Vehicle Maintenance Knowledge Base
# ---------------------------------------------------------------------------

MAINTENANCE_DOCS = [
    # Engine & Oil
    """ENGINE OIL CHANGE GUIDELINES
    Regular oil changes are critical for engine longevity. For fleet vehicles:
    - Petrol engines: change every 8,000-10,000 km or 6 months (whichever comes first).
    - Diesel engines: change every 6,000-8,000 km or 4 months due to higher soot.
    - Synthetic oil extends intervals to 12,000-15,000 km.
    Signs of overdue oil change: dark/gritty oil, engine noise, reduced fuel efficiency,
    oil warning light. Neglecting oil changes leads to sludge buildup, increased wear,
    and potential engine seizure. Always use manufacturer-recommended viscosity grade.""",

    """ENGINE MAINTENANCE — GENERAL
    Routine engine checks should include: coolant level and condition, serpentine belt
    tension and cracking, spark plug condition (petrol), injector cleaning (diesel),
    air filter inspection, and exhaust system checks. High-mileage vehicles (>100,000 km)
    require closer monitoring of gasket integrity, compression tests, and timing
    chain/belt replacement. Unusual noises (knocking, ticking) warrant immediate
    inspection. Engine overheating must be addressed immediately to prevent head
    gasket failure or block warping.""",

    # Brakes
    """BRAKE SYSTEM MAINTENANCE
    Brake inspection should occur every 20,000 km or when symptoms appear:
    - Squealing or grinding noises during braking.
    - Vibration or pulsation in the brake pedal.
    - Increased stopping distance.
    - Vehicle pulling to one side during braking.
    Brake pads: replace when thickness is below 3mm. Brake rotors: resurface or
    replace when below minimum thickness or if scoring is present. Brake fluid:
    flush every 2 years or 40,000 km — it absorbs moisture over time, reducing
    boiling point. For fleet trucks and buses, check air brake systems, slack
    adjusters, and compressor function regularly. Worn brakes are a critical safety
    hazard and must be addressed immediately.""",

    # Tires
    """TIRE MAINTENANCE AND REPLACEMENT
    Tire inspections should cover tread depth, pressure, alignment, and rotation:
    - Minimum legal tread depth: 1.6mm, but replace at 3mm for safety.
    - Check tire pressure monthly; under-inflation increases fuel consumption by 3-5%
      and causes uneven wear.
    - Rotate tires every 8,000-10,000 km to ensure even wear.
    - Wheel alignment check every 20,000 km or after hitting potholes/curbs.
    - Replace tires older than 6 years regardless of tread depth (rubber degrades).
    For fleet vehicles, mismatched tires affect handling and braking. All-season vs.
    seasonal tires should be selected based on operating climate. Worn-out tires
    significantly increase accident risk in wet conditions.""",

    # Battery
    """BATTERY CARE AND REPLACEMENT
    Vehicle battery lifespan is typically 3-5 years. Warning signs of a failing battery:
    - Slow engine cranking or clicking sound on start.
    - Dim headlights at idle.
    - Battery warning light on dashboard.
    - Swollen or corroded battery case.
    Preventive measures: clean terminals every 6 months, ensure tight connections,
    test voltage regularly (healthy battery: 12.4-12.7V when off, 13.7-14.7V running).
    Extreme temperatures accelerate degradation. For fleet vehicles, maintain a
    replacement schedule and keep jump-start equipment available. Electric/hybrid
    vehicles require specialized high-voltage battery diagnostics.""",

    # Transmission
    """TRANSMISSION SERVICE GUIDELINES
    Automatic transmission: fluid change every 50,000-80,000 km.
    Manual transmission: fluid change every 50,000-60,000 km.
    Signs of transmission issues: delayed or rough shifting, slipping gears, fluid
    leaks (red/brown fluid), burning smell, check engine light.
    Transmission fluid should be checked monthly — it should be red/pink and not
    smell burnt. Dark or burnt-smelling fluid indicates overheating damage.
    For fleet trucks with heavy loads, consider more frequent service intervals.
    Transmission failure is one of the most expensive repairs; preventive
    maintenance is far more cost-effective than replacement.""",

    # Coolant
    """COOLANT SYSTEM MAINTENANCE
    Engine coolant prevents overheating and freezing, and inhibits corrosion.
    - Coolant flush: every 40,000-60,000 km or 2-3 years.
    - Check coolant level weekly for fleet vehicles.
    - Inspect hoses for cracks, bulging, or soft spots quarterly.
    - Test coolant concentration with a refractometer (should be 50/50 mix).
    - Thermostat replacement if engine runs cold or overheats intermittently.
    - Radiator cap: replace if not holding pressure (causes boil-over).
    Overheating can cause catastrophic engine damage within minutes. If temperature
    gauge enters red zone, stop immediately and let the engine cool.""",

    # Fleet scheduling
    """FLEET MAINTENANCE SCHEDULING — BEST PRACTICES
    Effective fleet maintenance follows a tiered schedule:
    - Daily: driver walk-around (lights, tires, fluids, visible damage).
    - Weekly: fluid top-ups, tire pressure, windshield washer.
    - Monthly: battery check, brake visual inspection, belt inspection.
    - Quarterly: comprehensive inspection (all systems).
    - Annually: full service including transmission, coolant, brake fluid flush.
    Mileage-based: minor service every 10,000 km, major service every 40,000 km.
    Track each vehicle's service history in a centralized system. Vehicles with
    higher-than-average repair frequency should be flagged for potential retirement.
    Downtime cost often exceeds repair cost — preventive maintenance reduces both.""",

    # Emergency protocols
    """EMERGENCY MAINTENANCE PROTOCOLS
    Critical issues requiring immediate vehicle removal from service:
    - Brake failure or significant brake degradation.
    - Steering system malfunction.
    - Tire blowout or tread separation.
    - Engine overheating (temperature in red zone).
    - Fluid leaks creating fire hazard (fuel, oil on hot components).
    - Suspension failure affecting vehicle control.
    - Electrical fire or burning smell from wiring.
    Response procedure: 1) Remove vehicle from service immediately. 2) Document
    symptoms and conditions. 3) Tow to maintenance facility (do not drive).
    4) Perform root cause analysis. 5) Verify repair before returning to service.
    Never allow a vehicle with critical safety issues to remain in operation.""",

    # Fuel efficiency
    """FUEL EFFICIENCY OPTIMIZATION
    Poor fuel efficiency often indicates underlying maintenance issues:
    - Dirty air filter reduces efficiency by up to 10%.
    - Under-inflated tires increase rolling resistance and fuel use by 3-5%.
    - Misaligned wheels cause drag and uneven tire wear.
    - Worn spark plugs (petrol) cause incomplete combustion.
    - Clogged fuel injectors reduce atomization and power.
    - Dragging brakes (stuck caliper) waste energy as heat.
    Fleet-wide fuel tracking helps identify vehicles needing attention. A sudden
    drop in fuel efficiency (>15%) for a specific vehicle warrants immediate
    inspection. Regular maintenance keeps fuel costs predictable and minimizes
    environmental impact.""",

    # Seasonal maintenance
    """SEASONAL MAINTENANCE CHECKLIST
    Pre-summer: Check A/C system refrigerant and operation, coolant concentration,
    battery (heat accelerates degradation), tire pressure (increases with heat),
    cabin air filter.
    Pre-winter: Battery load test, antifreeze concentration, tire tread depth
    (consider winter tires), wiper blades, heater/defroster operation, door seal
    condition, emergency kit (blankets, torch, sand/salt).
    Monsoon/rainy season: Wiper condition, tire tread depth, headlight alignment,
    brake inspection (wet conditions demand more), rust-proofing inspection,
    drainage channel clearance.
    Seasonal transitions are when breakdowns spike — proactive checks reduce
    unplanned downtime by up to 40%.""",

    # High-mileage vehicles
    """HIGH-MILEAGE VEHICLE MANAGEMENT (>100,000 KM)
    Vehicles exceeding 100,000 km require enhanced monitoring:
    - Compression test to assess engine health.
    - Transmission fluid and filter change if not done recently.
    - Suspension bushings and shock absorbers — often worn by this mileage.
    - Water pump replacement (often done with timing belt).
    - CV joints/boots inspection (front-wheel drive).
    - Exhaust system inspection for rust and leaks.
    - Power steering fluid flush.
    Consider total cost of ownership: if annual maintenance exceeds 50% of the
    vehicle's value, replacement may be more economical. High-mileage fleet vehicles
    should be evaluated for reliability impact on operations.""",

    # Safety compliance
    """VEHICLE SAFETY COMPLIANCE AND INSPECTION
    Fleet vehicles must meet safety standards at all times:
    - Functional headlights, tail lights, brake lights, and indicators.
    - Working horn and windshield wipers.
    - Seat belts in proper condition for all seating positions.
    - Mirrors: properly adjusted and undamaged.
    - Fire extinguisher (commercial vehicles): charged and accessible.
    - First aid kit: stocked and not expired.
    - Vehicle registration and insurance documentation current.
    Regular safety audits (monthly for commercial fleets) ensure compliance and
    reduce liability. Document all inspections with date, inspector, and findings.
    Non-compliant vehicles must be removed from service until issues are resolved.""",

    # Warranty considerations
    """WARRANTY AND SERVICE RECORD MANAGEMENT
    Maintaining accurate service records is essential for warranty claims and resale:
    - Record all maintenance with date, mileage, work performed, and parts used.
    - Keep receipts for all services and parts purchases.
    - Follow manufacturer-recommended service intervals to maintain warranty.
    - Warranty typically covers powertrain for 5 years/100,000 km.
    - Extended warranties may cover additional components.
    - Warranty may be voided by: unauthorized modifications, missed scheduled
      services, use of non-approved fluids or parts.
    For fleet managers: centralized digital records enable trend analysis, cost
    tracking, and resale value optimization. Expired warranty vehicles should be
    evaluated for self-insurance vs. extended warranty purchase.""",
]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

_rag_instance = None


class MaintenanceRAG:
    """FAISS-backed retrieval of maintenance guidelines."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        docs = [
            Document(page_content=text, metadata={"source": f"guideline_{i}"})
            for i, text in enumerate(MAINTENANCE_DOCS)
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content.strip() for doc in results]


def get_rag() -> MaintenanceRAG:
    """Singleton accessor so the vector store is built only once."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MaintenanceRAG()
    return _rag_instance
