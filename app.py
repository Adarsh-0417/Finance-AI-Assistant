"""
app.py
------
Finance AI Assistant — Streamlit UI
Conversational finance chatbot with RAG over a built-in knowledge base,
SIP calculator, quick-action buttons, and guardrails.
"""

import os
import math
import logging
import streamlit as st

st.set_page_config(
    page_title="Finance AI Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background: #060912;
    color: #e2e8f0;
}

/* ── Header ── */
.fin-header {
    background: linear-gradient(135deg, #060912 0%, #0d1b2a 55%, #091822 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.fin-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 0%, rgba(56,189,248,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.fin-header h1 { color:#f0f9ff; font-size:1.75rem; font-weight:700; margin:0; letter-spacing:-0.03em; }
.fin-header p  { color:rgba(148,163,184,0.7); margin:0.25rem 0 0; font-size:0.88rem; }
.fin-ticker    { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#38bdf8; margin-top:0.5rem; opacity:0.6; }

/* ── Chat bubbles ── */
.msg-user {
    background: rgba(56,189,248,0.08);
    border-left: 3px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1rem;
    margin: 0.5rem 0;
    color: #bae6fd;
    font-size: 0.9rem;
}
.msg-bot {
    background: rgba(16,185,129,0.06);
    border-left: 3px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1rem;
    margin: 0.5rem 0;
    color: #d1fae5;
    font-size: 0.9rem;
    line-height: 1.7;
}
.msg-label {
    font-size:0.65rem; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; margin-bottom:0.3rem; opacity:0.5;
    font-family:'JetBrains Mono',monospace;
}
.msg-user .msg-label { color:#38bdf8; }
.msg-bot  .msg-label { color:#10b981; }

/* ── Quick action buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.8rem !important;
    transition: all 0.18s ease !important;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 0.5rem 0.85rem;
    font-size: 0.75rem;
    color: #fcd34d;
    margin-top: 0.5rem;
}

/* ── Calculator card ── */
.calc-card {
    background: #0d1b2a;
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.calc-result {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
    text-align: center;
    margin: 0.5rem 0 0.25rem;
}
.calc-sub {
    font-size: 0.78rem; color: #64748b;
    text-align: center; margin-bottom: 0.5rem;
}

/* ── Model pill ── */
.model-pill {
    display:inline-block;
    background:rgba(56,189,248,0.1);
    border:1px solid rgba(56,189,248,0.3);
    border-radius:20px; padding:3px 12px;
    font-size:0.7rem; color:#7dd3fc;
    font-family:'JetBrains Mono',monospace;
    margin-bottom:0.4rem;
}

/* ── Status badges ── */
.badge-ready { background:#064e3b; color:#6ee7b7; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; }
.badge-warn  { background:#431407; color:#fb923c; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; }

/* ── Input ── */
.stTextInput input {
    background: #0d1b2a !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 10px !important;
}
div[data-testid="stForm"] { border:none !important; padding:0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #08101a !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN FINANCIAL KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

FINANCE_KB = [
    # SIP & Mutual Funds
    "SIP stands for Systematic Investment Plan. It allows investors to invest a fixed amount in mutual funds at regular intervals (monthly/weekly). SIP helps in rupee cost averaging and building wealth over time through compounding.",
    "Mutual funds pool money from many investors to invest in diversified portfolios of stocks, bonds, or other securities. They are managed by professional fund managers.",
    "Types of mutual funds: Equity funds (invest in stocks, higher risk/return), Debt funds (invest in bonds, lower risk), Hybrid funds (mix of equity and debt), Index funds (track a market index like Nifty 50).",
    "ELSS (Equity Linked Savings Scheme) mutual funds offer tax deduction under Section 80C with a 3-year lock-in period. They are diversified equity funds with tax benefits.",
    "NAV (Net Asset Value) is the per-unit price of a mutual fund. It is calculated as: (Total Assets - Total Liabilities) / Number of Units.",

    # Investing basics
    "Diversification means spreading investments across different asset classes (equity, debt, gold, real estate) to reduce risk. 'Don't put all eggs in one basket.'",
    "Risk and return are directly related in investing. Higher potential returns generally come with higher risk. Conservative investors prefer debt/FDs; aggressive investors prefer equities.",
    "Compounding is earning returns on your returns. Starting early dramatically increases wealth. Example: Rs.1 lakh at 12% p.a. for 30 years becomes over Rs.29 lakhs.",
    "Asset allocation is dividing your portfolio among different asset categories. A common rule: subtract your age from 100 to get the percentage to invest in equities.",
    "Index funds track a market index (e.g., Nifty 50, Sensex) passively with very low expense ratios. They are ideal for long-term investors who want market returns.",
    "Stocks represent ownership in a company. Stock prices fluctuate based on company performance, market sentiment, and economic factors. Long-term equity investing historically beats inflation.",
    "PPF (Public Provident Fund) is a government-backed savings scheme with 15-year lock-in offering ~7.1% p.a. (tax-free) and tax deduction under Section 80C.",
    "Fixed Deposits (FDs) offer guaranteed returns at a fixed interest rate. They are low-risk but returns may not beat inflation over the long term.",

    # Personal Finance
    "The 50-30-20 budgeting rule: allocate 50% of income to needs (rent, food, utilities), 30% to wants (entertainment, dining out), and 20% to savings and investments.",
    "An emergency fund should cover 3-6 months of monthly expenses. Keep it in a liquid account like a savings account or liquid mutual fund.",
    "Term life insurance provides coverage for a specific period. It pays a death benefit if the insured dies during the term. It is the most affordable form of life insurance.",
    "Health insurance covers medical expenses. Always have adequate health insurance before starting to invest. Aim for a cover of at least Rs.5-10 lakhs.",
    "Credit score (CIBIL score) ranges from 300 to 900. A score above 750 is considered good and helps get loans at lower interest rates. Pay EMIs and credit card bills on time.",
    "Avoid high-interest debt like credit card debt (24-36% p.a.) and personal loans. Pay off such debt before investing as no investment consistently beats such high interest rates.",

    # Economic Concepts
    "Inflation is the rate at which the general price level of goods and services rises, eroding purchasing power. In India, target inflation is around 4%. Your investments must beat inflation to grow real wealth.",
    "Interest rate and bond prices move inversely. When RBI raises interest rates, existing bond prices fall; when rates fall, bond prices rise.",
    "GDP (Gross Domestic Product) is the total value of goods and services produced in a country. It is a key indicator of economic health.",
    "Recession is a period of significant economic decline (negative GDP growth for two consecutive quarters). During recessions, stock markets typically decline but can be good buying opportunities.",
    "Rupee cost averaging: by investing a fixed amount regularly (SIP), you buy more units when prices are low and fewer when prices are high, reducing average cost over time.",
    "XIRR (Extended Internal Rate of Return) measures actual returns on investments with irregular cash flows (like SIPs). It is more accurate than simple annualized returns for SIPs.",

    # Taxes
    "LTCG (Long Term Capital Gains) tax on equity: gains above Rs.1 lakh per year from equity/equity mutual funds held for over 1 year are taxed at 10%.",
    "STCG (Short Term Capital Gains) tax on equity: gains from equity sold within 1 year are taxed at 15%.",
    "Section 80C allows tax deduction up to Rs.1.5 lakh per year for investments in PPF, ELSS, NPS, EPF, life insurance premiums, etc.",
    "HRA (House Rent Allowance) exemption can reduce taxable income for salaried individuals who pay rent.",
]


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

defaults = {
    "chat_history"    : [],
    "rag_pipeline"    : None,
    "vectorstore"     : None,
    "embedding_model" : None,
    "active_model_key": "",
    "prefill_query"   : "",
    "initialized"     : False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# LAZY INIT — build KB index once
# ══════════════════════════════════════════════════════════════════════════════

def initialize_pipeline(model_key, temperature, max_new_tokens, embed_model,
                         top_k, score_threshold, rerank, rerank_top_n):
    """Build FAISS index from built-in KB and load the LLM."""
    from langchain_core.documents import Document
    from embeddings   import load_embedding_model, build_faiss_index
    from llm          import load_huggingface_llm
    from rag_pipeline import RAGPipeline

    with st.spinner("🔢 Loading embedding model…"):
        emb = load_embedding_model(embed_model)
        st.session_state.embedding_model = emb

    docs = [Document(page_content=chunk, metadata={"source": "FinanceKB", "page": i + 1})
            for i, chunk in enumerate(FINANCE_KB)]

    with st.spinner("🗄️ Building knowledge index…"):
        vs = build_faiss_index(docs, emb, index_dir="finance_faiss_index")
        st.session_state.vectorstore = vs

    with st.spinner(f"🤖 Loading **{model_key}** (first run downloads weights)…"):
        hf = load_huggingface_llm(model_key=model_key, temperature=temperature,
                                   max_new_tokens=max_new_tokens)

    pipeline = RAGPipeline(
        vectorstore=vs,
        llm=hf["llm"],
        task=hf["task"],
        model_id=hf["model_id"],
        top_k=top_k,
        score_threshold=score_threshold,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
    )

    # Inject finance-specific system persona into pipeline
    pipeline._finance_mode = True

    st.session_state.rag_pipeline    = pipeline
    st.session_state.active_model_key = model_key
    st.session_state.initialized      = True


# ══════════════════════════════════════════════════════════════════════════════
# QUERY WRAPPER — adds finance system prompt context
# ══════════════════════════════════════════════════════════════════════════════

FINANCE_SYSTEM_CONTEXT = (
    "[SYSTEM: You are a helpful, friendly Finance AI Assistant. "
    "Provide clear, simple, accurate financial explanations. "
    "Use bullet points for lists and structure your answers. "
    "Keep explanations beginner-friendly. "
    "If you are unsure, say so — never hallucinate. "
    "Do NOT give risky or misleading financial advice. "
    "Always add a short disclaimer for investment queries.]\n\n"
)

def run_query(question: str) -> str:
    pipeline = st.session_state.rag_pipeline
    if pipeline is None:
        return "⚠️ Assistant is not initialized yet. Click **Initialize** in the sidebar."

    augmented = FINANCE_SYSTEM_CONTEXT + question
    result = pipeline.query(augmented)
    answer = result.get("answer", "").strip()

    if not answer:
        answer = "I'm not sure about that. Could you rephrase or ask something more specific?"
    return answer


# ══════════════════════════════════════════════════════════════════════════════
# SIP CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def sip_maturity(monthly: float, rate_annual: float, years: int) -> tuple:
    """Returns (maturity_value, total_invested, total_gains)."""
    n   = years * 12
    r   = rate_annual / 100 / 12
    if r == 0:
        maturity = monthly * n
    else:
        maturity = monthly * (((1 + r) ** n - 1) / r) * (1 + r)
    invested = monthly * n
    return round(maturity), round(invested), round(maturity - invested)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("### 🤖 Language Model")
    from llm import get_model_display_names, MODEL_CATALOGUE
    model_names = get_model_display_names()
    model_key   = st.selectbox("Model", options=model_names, index=0,
                               help="Larger = better answers, more RAM needed.")
    cfg = MODEL_CATALOGUE[model_key]
    st.markdown(f'<div class="model-pill">~{cfg["ram_gb"]} GB RAM · {cfg["task"]}</div>',
                unsafe_allow_html=True)
    st.caption(f"`{cfg['id']}`")

    temperature    = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_new_tokens = st.slider("Max tokens", 128, 1024, 512, 64)

    st.divider()
    st.markdown("### 🔢 Embeddings")
    embed_model = st.selectbox("Sentence-transformer", options=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ])

    st.divider()
    st.markdown("### 🔍 Retrieval")
    top_k           = st.slider("Top-K chunks", 1, 10, 4)
    score_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.20, 0.05)
    use_rerank      = st.toggle("Cross-encoder re-ranking", value=False,
                                help="More accurate, slower.")
    rerank_top_n    = st.slider("Chunks after re-rank", 1, 6, 3, disabled=not use_rerank)

    st.divider()

    init_btn = st.button(
        "⚡ Initialize Assistant",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.initialized,
    )

    if init_btn:
        initialize_pipeline(model_key, temperature, max_new_tokens, embed_model,
                             top_k, score_threshold, use_rerank, rerank_top_n)
        st.rerun()

    if st.session_state.initialized:
        st.markdown(f'<span class="badge-ready">● READY</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="model-pill">🤖 {st.session_state.active_model_key}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">○ NOT INITIALIZED</span>', unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.reset_memory()
        st.rerun()

    st.divider()
    st.markdown(
        '<div class="disclaimer">⚠️ This assistant is for <b>educational purposes only</b>, '
        'not professional financial advice. Always consult a certified financial advisor '
        'before making investment decisions.</div>',
        unsafe_allow_html=True,
    )
    st.caption("🔒 100% local — no data leaves your machine")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="fin-header">
  <h1>💹 Finance AI Assistant</h1>
  <p>Your intelligent guide to personal finance, investing & financial concepts</p>
  <div class="fin-ticker">SIP · MUTUAL FUNDS · BUDGETING · INFLATION · STOCKS · TAX SAVING</div>
</div>
""", unsafe_allow_html=True)

chat_col, tools_col = st.columns([3, 2], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — CHAT
# ══════════════════════════════════════════════════════════════════════════════

with chat_col:
    st.markdown("### 💬 Chat")

    # Quick action buttons
    st.markdown("**Quick questions:**")
    btn_row1 = st.columns(4)
    quick_labels = [
        ("📈 Explain SIP",        "Explain SIP (Systematic Investment Plan) in simple terms with an example."),
        ("💰 How to save money?", "Give me practical tips on how to save money with a beginner-friendly savings plan."),
        ("📊 What is inflation?", "What is inflation, how does it affect my savings, and how can I beat it?"),
        ("🎓 Investment basics",  "Explain the basics of investing for a complete beginner. Where should I start?"),
    ]
    for col, (label, query) in zip(btn_row1, quick_labels):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.prefill_query = query

    btn_row2 = st.columns(4)
    quick_labels2 = [
        ("🏦 Mutual Funds 101",   "Explain mutual funds, their types, and how to choose the right one."),
        ("🧮 50-30-20 Rule",      "Explain the 50-30-20 budgeting rule with an example for someone earning ₹50,000/month."),
        ("📉 Market risk",        "What is investment risk? How do I manage it through diversification?"),
        ("💳 Credit score tips",  "How can I improve my credit score? What factors affect it?"),
    ]
    for col, (label, query) in zip(btn_row2, quick_labels2):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.prefill_query = query

    st.markdown("")

    # Chat history display
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="color:#475569;font-size:0.88rem;text-align:center;padding-top:3rem;">'
                '👋 Hi! Ask me anything about personal finance, investments, or budgeting.<br>'
                '<span style="font-size:0.78rem;opacity:0.6;">Initialize the assistant from the sidebar first.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="msg-user"><div class="msg-label">You</div>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="msg-bot"><div class="msg-label">Finance AI</div>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        default_val = st.session_state.pop("prefill_query", "") if "prefill_query" in st.session_state else ""
        user_input = st.text_input(
            "Message",
            value=default_val,
            placeholder="Ask about budgeting, SIP, inflation, tax saving, stocks…",
            label_visibility="collapsed",
        )
        send = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    if send and user_input.strip():
        if not st.session_state.initialized:
            st.warning("⚠️ Please initialize the assistant from the sidebar first.")
        else:
            with st.spinner("Thinking…"):
                answer = run_query(user_input.strip())
            st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant",  "content": answer})
            st.rerun()

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">⚠️ Educational purposes only — not professional financial advice.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — TOOLS
# ══════════════════════════════════════════════════════════════════════════════

with tools_col:

    # ── SIP Calculator ─────────────────────────────────────────────────────────
    st.markdown("### 🧮 SIP Calculator")
    st.markdown('<div class="calc-card">', unsafe_allow_html=True)

    monthly_amt  = st.number_input("Monthly Investment (₹)", min_value=500, max_value=500000,
                                    value=5000, step=500)
    annual_rate  = st.slider("Expected Annual Return (%)", min_value=4.0, max_value=20.0,
                              value=12.0, step=0.5)
    years        = st.slider("Investment Period (Years)", min_value=1, max_value=40, value=10)

    maturity, invested, gains = sip_maturity(monthly_amt, annual_rate, years)

    def fmt(n): return f"₹{n:,.0f}"

    st.markdown(f'<div class="calc-result">{fmt(maturity)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="calc-sub">Estimated Maturity Value</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Invested", fmt(invested))
    with c2:
        st.metric("Estimated Gains", fmt(gains), delta=f"+{round((gains/invested)*100, 1)}%")

    st.caption(
        "⚠️ Returns are not guaranteed. Actual returns vary. "
        "Past performance doesn't guarantee future results."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Savings Growth Calculator ──────────────────────────────────────────────
    st.markdown("### 💰 Savings Growth Calculator")
    st.markdown('<div class="calc-card">', unsafe_allow_html=True)

    initial_saving = st.number_input("Initial Amount (₹)", min_value=0, max_value=10000000,
                                      value=10000, step=1000)
    monthly_saving = st.number_input("Monthly Addition (₹)", min_value=0, max_value=100000,
                                      value=2000, step=500)
    fd_rate        = st.slider("Interest Rate (% p.a.)", min_value=1.0, max_value=15.0,
                                value=7.0, step=0.25)
    fd_years       = st.slider("Duration (Years)", min_value=1, max_value=30, value=5)

    r_m = fd_rate / 100 / 12
    n_m = fd_years * 12
    fv_lump = initial_saving * (1 + r_m) ** n_m
    fv_sip  = monthly_saving * (((1 + r_m) ** n_m - 1) / r_m) * (1 + r_m) if r_m else monthly_saving * n_m
    total_fv = round(fv_lump + fv_sip)
    total_in = initial_saving + monthly_saving * n_m

    st.markdown(f'<div class="calc-result">{fmt(total_fv)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="calc-sub">Total Savings Value</div>', unsafe_allow_html=True)

    cs1, cs2 = st.columns(2)
    with cs1:
        st.metric("Total Put In", fmt(total_in))
    with cs2:
        gain_s = total_fv - total_in
        st.metric("Interest Earned", fmt(gain_s))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Finance Tips ───────────────────────────────────────────────────────────
    with st.expander("📚 Key Finance Principles"):
        st.markdown("""
**🏦 Start Early**  
Time in market > timing the market. Start investing even with small amounts.

**📊 50-30-20 Rule**  
- 50% → Needs (rent, food, bills)  
- 30% → Wants (eating out, entertainment)  
- 20% → Savings & Investments

**🛡️ Emergency Fund First**  
Keep 3-6 months of expenses in liquid savings before investing.

**📈 SIP for Discipline**  
Automate monthly investments via SIP to avoid emotional decisions.

**🔄 Diversify**  
Spread across equity, debt, and gold. Don't put all eggs in one basket.
        """)

    with st.expander("ℹ️ How This Assistant Works"):
        st.markdown("""
**RAG Pipeline:**
1. Built-in financial knowledge base (30+ concepts)
2. Your query → FAISS semantic search → relevant chunks
3. Relevant context + chat history → LLM → answer

**Memory:** The assistant remembers your conversation context within a session.

**Models available:** TinyLlama, Phi-2, Phi-3-mini, Flan-T5  
All run 100% locally on your CPU.
        """)
