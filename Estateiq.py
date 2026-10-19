import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv()

st.set_page_config(page_title="EstateIQ", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0A0F1E; color: #E8EDF5; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0D1526 !important; border-right: 1px solid #1E2D45; }
[data-testid="stSidebar"] label { color: #7A9BBF !important; font-size: 0.8rem !important; }
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    background: #1A2332 !important; border: 1px solid #2A3D57 !important;
    color: #E8EDF5 !important; border-radius: 8px !important;
}
.stSlider [data-testid="stSlider"] > div > div { background: #C9A84C !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C, #A67C2E) !important;
    color: #0A0F1E !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    padding: 10px 0 !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Cards */
.card {
    background: #1A2332; border: 1px solid #1E2D45;
    border-radius: 12px; padding: 18px; margin-bottom: 14px;
}
.card.gold  { border-left: 3px solid #C9A84C; }
.card.blue  { border-left: 3px solid #4FC3F7; }
.card.green { border-left: 3px solid #4CAF50; }
.card.red   { border-left: 3px solid #EF5350; }
.card.purple{ border-left: 3px solid #AB47BC; }

/* Metric row */
.metric { background: #1A2332; border: 1px solid #1E2D45; border-radius: 10px; padding: 14px 18px; }
.metric-val { font-size: 1.6rem; font-weight: 800; color: #C9A84C; }
.metric-lbl { font-size: 0.7rem; color: #5A7A9A; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* Pipeline tracker */
.pipe-wrap { display: flex; align-items: center; gap: 0; margin: 18px 0; }
.pipe-step {
    flex: 1; text-align: center; font-size: 0.68rem; font-weight: 600;
    padding: 8px 4px; background: #1A2332; color: #4A6A8A;
    border-top: 1px solid #1E2D45; border-bottom: 1px solid #1E2D45;
    border-right: 1px solid #1E2D45; position: relative;
}
.pipe-step:first-child { border-left: 1px solid #1E2D45; border-radius: 8px 0 0 8px; }
.pipe-step:last-child  { border-radius: 0 8px 8px 0; }
.pipe-step.done   { background: #0D2818; color: #4CAF50; border-color: #1E4A2A; }
.pipe-step.active { background: #1A2A0D; color: #C9A84C; border-color: #3A4A1A; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.6} }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1E2D45; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #5A7A9A !important; font-size: 0.8rem !important; border: none !important; }
.stTabs [aria-selected="true"] { color: #C9A84C !important; border-bottom: 2px solid #C9A84C !important; }
.stTabs [data-baseweb="tab-panel"] { background: #1A2332; border: 1px solid #1E2D45; border-top: none; border-radius: 0 0 10px 10px; padding: 20px; }

/* Output text */
.out { font-size: 0.82rem; line-height: 1.8; color: #B8CDE0; white-space: pre-wrap; }
.tag { display:inline-block; background:rgba(201,168,76,0.15); color:#C9A84C; border:1px solid rgba(201,168,76,0.3); font-size:0.65rem; font-weight:700; padding:2px 9px; border-radius:12px; letter-spacing:.8px; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

#Shared State 
class State(TypedDict):
    location: str; budget: str; prop_type: str; purpose: str; notes: str
    market: str; properties: str; finance: str; legal: str; negotiation: str; report: str

#Agents
def run_agent(llm, role, instruction, context):
    return llm.invoke(f"You are a real estate {role}.\n\n{instruction}\n\nContext:\n{context}").content

def market_agent(s, llm):
    out = run_agent(llm, "Market Analyst",
        f"Analyse the {s['prop_type']} market in {s['location']} for {s['purpose']}. Budget: {s['budget']}.\n"
        "Cover: 1) Current conditions 2) Price trends 3) Best micro-markets 4) Risks 5) Market score /10.",
        s['notes'] or "No extra notes.")
    return {"market": out}

def property_agent(s, llm):
    out = run_agent(llm, "Property Scout",
        f"Recommend 3 specific {s['prop_type']} options in {s['location']}. Budget: {s['budget']}. Purpose: {s['purpose']}.\n"
        "For each: area, price, specs, 2 pros, 1 con, match score /10.",
        s['market'][:400])
    return {"properties": out}

def finance_agent(s, llm):
    out = run_agent(llm, "Financial Advisor",
        f"Financial analysis for {s['prop_type']} in {s['location']}. Budget: {s['budget']}. Purpose: {s['purpose']}.\n"
        "Cover: EMI estimate, rental yield %, ROI over 5yr, total acquisition cost, investment grade A–D.",
        s['market'][:300])
    return {"finance": out}

def legal_agent(s, llm):
    out = run_agent(llm, "Legal Advisor",
        f"Legal checklist for buying {s['prop_type']} in {s['location']}.\n"
        "Cover: title check, RERA compliance, key documents, top 3 red flags to watch.",
        s['purpose'])
    return {"legal": out}

def negotiation_agent(s, llm):
    out = run_agent(llm, "Negotiation Coach",
        f"Negotiation strategy for {s['prop_type']} in {s['location']}. Budget: {s['budget']}.\n"
        "Cover: opening offer %, 3 tactics, non-price concessions, walk-away rule.",
        s['market'][:250])
    return {"negotiation": out}

def report_agent(s, llm):
    out = run_agent(llm, "Chief Advisor",
        f"Write an executive report for: {s['prop_type']} in {s['location']}, budget {s['budget']}, purpose {s['purpose']}.\n"
        "Sections: Executive Summary | Top Recommendation | Financial Verdict | Legal Priority | Next Steps (5 actions).",
        f"Market: {s['market'][:300]}\nFinance: {s['finance'][:300]}\nLegal: {s['legal'][:200]}")
    return {"report": out}

# LangGraph 
@st.cache_resource
def build_graph(model, temp):
    llm = ChatGroq(model=model, temperature=temp)
    g = StateGraph(State)
    for name, fn in [("market", market_agent), ("properties", property_agent),
                     ("finance", finance_agent), ("legal", legal_agent),
                     ("negotiation", negotiation_agent), ("report", report_agent)]:
        g.add_node(name, lambda s, f=fn: f(s, llm))
    g.set_entry_point("market")
    for a, b in [("market","properties"),("properties","finance"),
                 ("finance","legal"),("legal","negotiation"),("negotiation","report")]:
        g.add_edge(a, b)
    g.add_edge("report", END)
    return g.compile()

# Sidebar 
with st.sidebar:
    st.markdown("<div style='padding:16px 0 10px'><span style='font-size:1.4rem;font-weight:800;color:#E8EDF5'>🏠 Estate<span style='color:#C9A84C'>IQ</span></span><br><span style='font-size:0.68rem;color:#3A5A7A;letter-spacing:1px;text-transform:uppercase'>AI Real Estate Team</span></div>", unsafe_allow_html=True)
    location  = st.text_input("Location", "Bandra West, Mumbai")
    prop_type = st.selectbox("Property Type", ["2BHK Apartment","3BHK Apartment","Villa","Studio","Penthouse","Commercial"])
    purpose   = st.selectbox("Purpose", ["End Use","Investment / Rental","Resale Flip"])
    col1, col2 = st.columns(2)
    bmin = col1.number_input("Min ₹L", 50, 50000, 150, 25)
    bmax = col2.number_input("Max ₹L", 50, 50000, 250, 25)
    model = st.selectbox("Model", ["llama-3.1-8b-instant","openai/gpt-oss-120b"])
    temp  = st.slider("Creativity", 0.0, 1.0, 0.3, 0.05)
    notes = st.text_area("Special Requirements", placeholder="School nearby, east-facing, sea view…", height=90)
    go    = st.button("🔍 Run AI Team", use_container_width=True)
    st.markdown("<hr style='border-color:#1E2D45;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#3A5A7A;line-height:2.2'>📊 Market Analyst<br>🏠 Property Scout<br>💹 Financial Advisor<br>⚖️ Legal Advisor<br>🤝 Negotiation Coach<br>📋 Report Compiler</div>", unsafe_allow_html=True)

#Header 
st.markdown("<h1 style='font-size:1.7rem;font-weight:800;margin:0'>Real Estate <span style='color:#C9A84C'>Intelligence</span> Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#3A5A7A;font-size:0.82rem;margin:4px 0 18px'>6 specialist AI agents · LangGraph pipeline · Market → Scout → Finance → Legal → Negotiate → Report</p>", unsafe_allow_html=True)

#Metrics 
r = st.session_state.get("result", {})
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"<div class='metric'><div class='metric-val'>6</div><div class='metric-lbl'>AI Agents</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric'><div class='metric-val'>360°</div><div class='metric-lbl'>Analysis</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric'><div class='metric-val'>{st.session_state.get('runs',0)}</div><div class='metric-lbl'>Reports Run</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric'><div class='metric-val' style='color:#4FC3F7'>{'✓ Ready' if r.get('report') else '—'}</div><div class='metric-lbl'>Status</div></div>", unsafe_allow_html=True)

#Pipeline tracker
STEPS   = ["📊 Market","🏠 Scout","💹 Finance","⚖️ Legal","🤝 Negotiate","📋 Report"]
KEYS    = ["market","properties","finance","legal","negotiation","report"]
current = st.session_state.get("step", -1)

def pipe_html(current, result):
    parts = []
    for i, label in enumerate(STEPS):
        done   = KEYS[i] in result and result[KEYS[i]]
        active = i == current
        cls    = "done" if done else ("active" if active else "")
        parts.append(f"<div class='pipe-step {cls}'>{label}</div>")
    return "<div class='pipe-wrap'>" + "".join(parts) + "</div>"

pipeline_slot = st.empty()
pipeline_slot.markdown(pipe_html(current, r), unsafe_allow_html=True)

#Run 
if go:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Add GROQ_API_KEY to your .env file.")
    else:
        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        st.session_state["result"] = {}

        budget = f"₹{bmin}L – ₹{bmax}L"
        s: State = dict(location=location, budget=budget, prop_type=prop_type,
                        purpose=purpose, notes=notes,
                        market="", properties="", finance="", legal="", negotiation="", report="")

        llm  = ChatGroq(model=model, temperature=temp)
        fns  = [market_agent, property_agent, finance_agent, legal_agent, negotiation_agent, report_agent]
        info = st.empty()

        for i, fn in enumerate(fns):
            st.session_state["step"] = i
            pipeline_slot.markdown(pipe_html(i, st.session_state["result"]), unsafe_allow_html=True)
            info.info(f"🤖 **{STEPS[i].split(' ',1)[1]}** is analysing… ({i+1}/6)")
            s.update(fn(s, llm))
            st.session_state["result"][KEYS[i]] = s[KEYS[i]]

        st.session_state["step"] = -1
        pipeline_slot.markdown(pipe_html(-1, st.session_state["result"]), unsafe_allow_html=True)
        info.success("✅ All 6 agents done! Scroll down to view results.")
        st.rerun()

#Results 
if r.get("report"):
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tabs = st.tabs(["📋 Final Report","📊 Market","🏠 Properties","💹 Finance","⚖️ Legal","🤝 Negotiation"])
    colors = ["gold","blue","green","blue","red","purple"]

    contents = [r.get("report",""), r.get("market",""), r.get("properties",""),
                r.get("finance",""), r.get("legal",""), r.get("negotiation","")]

    for tab, content, color in zip(tabs, contents, colors):
        with tab:
            st.markdown(f"<div class='card {color}'><div class='out'>{content}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download Report", r.get("report",""),
                           file_name=f"EstateIQ_{location.replace(' ','_')}.txt", use_container_width=True)
    with c2:
        full = "\n\n---\n\n".join([f"# {STEPS[i]}\n{r.get(KEYS[i],'')}" for i in range(6)])
        st.download_button("📄 Download Full Dossier", full,
                           file_name=f"EstateIQ_Full_{location.replace(' ','_')}.md",
                           mime="text/markdown", use_container_width=True)
elif not go:
    st.markdown("""
    <div style='background:#1A2332;border:1px dashed #1E2D45;border-radius:12px;padding:42px;text-align:center;margin-top:8px'>
        <div style='font-size:2rem;margin-bottom:12px'>🏠</div>
        <div style='font-size:1.1rem;font-weight:700;color:#E8EDF5;margin-bottom:8px'>Configure & run your AI team</div>
        <div style='font-size:0.8rem;color:#3A5A7A;max-width:360px;margin:0 auto;line-height:1.7'>
            Fill in property details on the left, then click <b style='color:#C9A84C'>Run AI Team</b>.<br>
            Six agents will analyse your query end-to-end.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:32px;text-align:center;font-size:0.68rem;color:#1E2D45;border-top:1px solid #1E2D45;padding-top:16px'>EstateIQ · LangGraph + Groq · For informational purposes only</div>", unsafe_allow_html=True)
