# 🏛️ EstateIQ — AI Real Estate Agent Team

> A 6-agent LangGraph pipeline that delivers 360° real estate intelligence — market analysis, property scouting, financial modelling, legal compliance, and negotiation strategy — served on a professional Streamlit dashboard.

---

## 🤖 Agent Architecture

```
[User Profile: Location + Budget + Type + Purpose]
              │
              ▼
    ┌─────────────────────┐
    │   Market Analyst    │  → Market trends, pricing, demand drivers, risk score
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Property Scout    │  → 4 matched property profiles with priority ranking
    └──────────┬──────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   Financial Advisor      │  → ROI, EMI, rental yield, tax implications
    └──────────┬───────────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Legal Advisor     │  → Due diligence, RERA, docs checklist, red flags
    └──────────┬──────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   Negotiation Coach      │  → Tactics, scripts, leverage, walk-away points
    └──────────┬───────────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Report Compiler   │  → Executive report + 7-day action plan
    └─────────────────────┘
```

---

## ✨ Features

- **6 Specialist AI Agents** running in a structured LangGraph pipeline
- **Live Agent Status Cards** — animated badges show which agent is currently working
- **360° Analysis** — market, property, finance, legal, and negotiation in one run
- **Professional Dark Dashboard** — luxury real estate aesthetic with gold accents
- **Tabbed Results** — each agent's output in its own tab for easy navigation
- **Download Reports** — export the full dossier as `.txt` or `.md`
- **Configurable** — choose model (`llama-3.3-70b` or `llama-3.1-8b`) and temperature
- **India-tuned** — budget in ₹ Lakhs, RERA-aware legal agent, Indian tax context

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agentic Framework** | LangGraph (StateGraph) |
| **LLM** | Groq — Llama 3.3 70B / Llama 3.1 8B |
| **LLM Integration** | `langchain-groq` |
| **UI** | Streamlit + Custom CSS (dark luxury theme) |
| **State** | `TypedDict` shared state across 6 agents |

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/estateiq-real-estate-agent.git
cd estateiq-real-estate-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🗂️ Agent State Schema

```python
class RealEstateState(TypedDict):
    # Inputs
    query: str           # Free-text client requirements
    location: str        # City/area
    budget: str          # Budget range in ₹ Lakhs
    property_type: str   # 2BHK, Villa, Commercial, etc.
    purpose: str         # End use, Investment, Resale, Lease

    # Agent outputs (populated sequentially)
    market_analysis: str
    property_recommendations: str
    financial_analysis: str
    legal_compliance: str
    negotiation_strategy: str
    final_report: str
```

---

## 📁 Project Structure

```
estateiq-real-estate-agent/
│
├── app.py              # Main app: 6 LangGraph agents + Streamlit UI
├── requirements.txt    # Python dependencies
└── .env                # GROQ_API_KEY (not committed)
```

---

## 📌 Usage

1. Enter **City / Area** (e.g. `Bandra West, Mumbai`)
2. Select **Property Type** and **Purpose**
3. Set your **Budget range** in ₹ Lakhs
4. Add any **special requirements** in the text box
5. Click **🔍 Analyse with AI Team**
6. Watch agents work in real-time via the status cards
7. Browse results in the 6 tabs
8. Download your full dossier

---

## ⚠️ Disclaimer

EstateIQ is an AI-powered research tool for informational purposes only. All analysis is LLM-generated and should not be treated as professional financial, legal, or investment advice. Always consult licensed professionals before making property decisions.

---

## 🙌 Built With

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Groq](https://groq.com) — ultra-fast LLM inference
- [Streamlit](https://streamlit.io) — rapid UI development
