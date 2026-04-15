import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))

from generator import recommend

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Craving-to-Order",
    page_icon="🍽️",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    /* Force white background globally */
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #f5f5f5; }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a !important;
    }
    .subtitle {
        font-size: 1rem;
        color: #555555 !important;
        margin-bottom: 1.5rem;
    }
    .response-box {
        background-color: #fafafa;
        border-left: 4px solid #ff6b35;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #1a1a1a !important;
    }
    .response-box strong { color: #1a1a1a !important; }
    .cost-box {
        background-color: #f0f7ff;
        border: 1px solid #cce0ff;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1a1a1a !important;
        margin-top: 0.8rem;
    }
    .hallucination-warning {
        background-color: #fff8e1;
        border: 1px solid #ffcc02;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1a1a1a !important;
        margin-top: 0.8rem;
    }
    .history-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #e0e0e0;
        font-size: 0.85rem;
        color: #333333 !important;
    }
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# --- LAYOUT ---
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    st.markdown('<div class="main-title">🍽️ Craving-to-Order</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Tell me what you\'re craving. I\'ll find the right dish.</div>', unsafe_allow_html=True)

    query = st.text_input(
        label="What are you craving?",
        placeholder="e.g. something creamy and mild under ₹300, spicy street food, light South Indian breakfast...",
        label_visibility="collapsed"
    )

    col_btn, col_fresh, col_examples = st.columns([1, 1, 2])
    with col_btn:
        submitted = st.button("Find dishes →", type="primary", use_container_width=True)
    with col_fresh:
        if st.button("Start fresh", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    with col_examples:
        st.markdown(
            "<small style='color:#888888'>Try: <em>sour and filling</em> · "
            "<em>dessert under 200</em> · <em>Kerala seafood</em> · <em>momos</em></small>",
            unsafe_allow_html=True
        )

    if submitted and query.strip():
        with st.spinner("Finding the right dishes for you..."):
            result = recommend(query.strip(), conversation_history=st.session_state.conversation_history)

        st.markdown("---")

        response_html = result["response"].replace("\n", "<br>")
        st.markdown(f'<div class="response-box">{response_html}</div>', unsafe_allow_html=True)

        if result["hallucination_flagged"]:
            st.markdown(
                f'<div class="hallucination-warning">⚠️ <strong>Internal flag:</strong> '
                f'{len(result["flagged_dishes"])} dish(es) not verified in menu data — '
                f'{", ".join(result["flagged_dishes"])}</div>',
                unsafe_allow_html=True
            )

        cost = result["cost"]
        st.markdown(
            f'<div class="cost-box">📊 This query — '
            f'Tokens: <strong>{cost["total_tokens"]}</strong> · '
            f'Cost: <strong>${cost["estimated_cost_usd"]:.6f}</strong> · '
            f'Breakdown: expansion <strong>{cost["breakdown"]["expansion_tokens"]}t</strong> · '
            f'embed <strong>{cost["breakdown"]["embed_tokens"]}t</strong> · '
            f'generation <strong>{cost["breakdown"]["generation_tokens"]}t</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.session_state.total_cost += cost["estimated_cost_usd"]
        st.session_state.total_queries += 1
        st.session_state.history.insert(0, query.strip())
        # Store turn in conversation history (keep last 3 only)
     dishes_recommended = [hit["dish"] for hit in result["hits"]]
cuisines_recommended = list(set([
    hit.get("cuisine_type", "") 
    for hit in result["hits"] 
    if hit.get("cuisine_type")
]))
st.session_state.conversation_history.append({
    "query": query.strip(),
    "dishes": dishes_recommended[:3],
    "cuisines": cuisines_recommended[:2]
})
if len(st.session_state.conversation_history) > 3:
    st.session_state.conversation_history.pop(0)

elif submitted and not query.strip():
    st.warning("Please enter a craving first.")

with col_sidebar:
    st.markdown("#### 📈 Session Stats")
    turns = len(st.session_state.conversation_history)
    if turns > 0:
        st.caption(f"Conversation: {turns} turn{'s' if turns > 1 else ''} in memory")
    st.metric("Queries this session", st.session_state.total_queries)
    st.metric("Total cost (USD)", f"${st.session_state.total_cost:.6f}")
    if st.session_state.total_queries > 0:
        avg = st.session_state.total_cost / st.session_state.total_queries
        st.metric("Avg cost per query", f"${avg:.6f}")

    st.markdown("---")
    st.markdown("#### 🕒 Query History")
    if st.session_state.history:
        for h in st.session_state.history[:10]:
            st.markdown(f'<div class="history-item">→ {h}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#aaa'>No queries yet</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
<small style='color:#333333'>
<strong>Craving-to-Order</strong> is a RAG-based food discovery layer over Delhi restaurant menus.<br><br>
• 17 restaurants · 674 dishes<br>
• Query expansion via GPT-4o-mini<br>
• Vector search via Pinecone<br>
• Filters: budget · diet · cuisine · taste<br>
• Hallucination detection on every query
</small>
""", unsafe_allow_html=True)
