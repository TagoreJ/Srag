import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sai Teachings · RAG Chatbot",
    page_icon="🪷",
    layout="centered",
)

# ── Beautiful White Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #fafaf8;
    min-height: 100vh;
}

header[data-testid="stHeader"] {
    background: transparent;
    border-bottom: none;
}

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f5f0eb 100%);
    border-bottom: 1px solid #e8e0d8;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(212,175,107,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(180,140,200,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-lotus {
    font-size: 2.8rem;
    margin-bottom: 0.5rem;
    display: block;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #2c1810;
    margin: 0 0 0.4rem;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.hero p {
    color: #8a7560;
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #d4af6b, #c9956a);
    margin: 1rem auto 0;
    border-radius: 2px;
}

/* ── Status Pill ── */
.status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    margin: 1rem auto;
    max-width: 320px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 50px;
    font-size: 0.8rem;
    color: #16a34a;
    font-weight: 500;
}
.status-dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.3); }
}

/* ── Index Info Badge ── */
.index-badge {
    background: #fff8f0;
    border: 1px solid #f0dfc0;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: #92651a;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 auto;
    font-weight: 500;
}

/* ── Chat Area ── */
.chat-area {
    max-width: 760px;
    margin: 0 auto;
    padding: 1.5rem 1rem 2rem;
}

/* ── Message Bubbles ── */
[data-testid="stChatMessage"] {
    padding: 0.3rem 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    background: linear-gradient(135deg, #2c1810 0%, #4a2c1a 100%);
    color: #fdf6ee;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.65;
    display: block;
    box-shadow: 0 2px 12px rgba(44,24,16,0.15);
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
    background: #ffffff;
    color: #2c1810;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
    display: block;
    border: 1px solid #ede6db;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #2c1810, #4a2c1a) !important;
    border-radius: 50% !important;
    color: #f0c67a !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #d4af6b, #c9956a) !important;
    border-radius: 50% !important;
}

/* ── Input Box ── */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1.5px solid #e0d5c8 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    padding: 0.2rem 0.5rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #d4af6b !important;
    box-shadow: 0 4px 20px rgba(212,175,107,0.2) !important;
}
[data-testid="stChatInput"] textarea {
    color: #2c1810 !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #b0a090 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ede6db !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif;
    color: #2c1810;
    font-size: 1.1rem;
}

/* ── Sidebar Metrics ── */
.metric-card {
    background: linear-gradient(135deg, #fdf6ee, #faf0e6);
    border: 1px solid #e8d8c0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #92651a;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #2c1810;
    font-family: 'Playfair Display', serif;
}

/* ── Welcome Card ── */
.welcome-card {
    background: linear-gradient(135deg, #fff8f0, #fdf4ec);
    border: 1px solid #e8d8c0;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem auto;
    max-width: 520px;
}
.welcome-card h3 {
    font-family: 'Playfair Display', serif;
    color: #2c1810;
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0 0 0.6rem;
}
.welcome-card p {
    color: #7a6555;
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0;
}
.sample-questions {
    margin-top: 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.sample-q {
    background: #ffffff;
    border: 1px solid #e0d0bc;
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    font-size: 0.82rem;
    color: #5a4535;
    cursor: pointer;
    text-align: left;
    font-style: italic;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #d4af6b !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #fafaf8; }
::-webkit-scrollbar-thumb { background: #d4c4b0; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #b8a090; }

/* ── Error box ── */
.error-box {
    background: #fff5f5;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #dc2626;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}

/* ── Source chips ── */
.source-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.source-chip {
    background: #f5efe6;
    border: 1px solid #e0d0b8;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: #7a5c35;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Load secrets ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_clients():
    groq_api_key    = st.secrets["GROQ_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    index_name      = st.secrets["PINECONE_INDEX_NAME"]

    pc    = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    groq  = Groq(api_key=groq_api_key)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    stats = index.describe_index_stats()
    return groq, index, model, index_name, stats


# ── RAG function ──────────────────────────────────────────────────────────────
def rag_chatbot(user_query: str, groq_client, pinecone_index, embedding_model, top_k: int = 3):
    # 1. Embed query
    query_embedding = embedding_model.encode(user_query).tolist()

    # 2. Query Pinecone
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    relevant_chunks = []
    source_files    = []
    for match in results.matches:
        if match.get("metadata"):
            text = match["metadata"].get("text", "")
            fname = match["metadata"].get("file_name", "")
            if text:
                relevant_chunks.append(text)
            if fname and fname not in source_files:
                source_files.append(fname)

    if not relevant_chunks:
        return "I could not find relevant teachings to answer your question.", []

    # 3. Build prompt
    system_message = (
        "You are an AI inspired by the teachings of Sri Sathya Sai Baba.\n\n"
        "STYLE:\n"
        "- Speak in a calm, kind, and simple tone\n"
        "- Be SHORT and clear (max 3–4 sentences)\n"
        "- Avoid long explanations or storytelling\n"
        "- Do NOT sound like a lecture\n\n"
        "RULES:\n"
        "- Answer ONLY using the given teachings\n"
        "- If answer not found, say: 'I do not find guidance in the provided teachings.'\n"
    )
    user_message = (
        f"Question:\n{user_query}\n\n"
        "Relevant teachings:\n"
        f"{chr(10).join(relevant_chunks[:2])}\n\n"
        "Answer briefly and gently."
    )

    # 4. Call Groq
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        model="llama-3.1-8b-instant",
    )
    return response.choices[0].message.content, source_files


# ── Init ──────────────────────────────────────────────────────────────────────
try:
    groq_client, pinecone_index, embedding_model, index_name, stats = init_clients()
    total_vectors = stats.total_vector_count
    ready = True
except Exception as e:
    ready = False
    init_error = str(e)


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-lotus">🪷</span>
    <h1>Sai Teachings</h1>
    <p>Ask • Seek • Receive Wisdom</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Status / Error ────────────────────────────────────────────────────────────
if ready:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center; margin: 0.8rem 0 0.3rem;">
            <div class="status-bar">
                <span class="status-dot"></span>
                Connected · {total_vectors:,} teachings indexed
            </div>
            <div class="index-badge">📚 Index: <strong>{index_name}</strong></div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="error-box">
        ⚠️ <strong>Could not connect.</strong> Please check your Streamlit secrets.<br>
        <code>GROQ_API_KEY</code>, <code>PINECONE_API_KEY</code>, <code>PINECONE_INDEX_NAME</code><br><br>
        Error: {init_error}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪷 About")
    st.markdown("""
    <div style="font-size:0.85rem; color:#5a4535; line-height:1.7;">
        This chatbot draws wisdom from the teachings of <strong>Sri Sathya Sai Baba</strong>,
        stored in a Pinecone vector database and powered by <strong>Groq</strong> (LLaMA 3.1).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Index Stats")
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Total Teachings</div>
        <div class="value">{total_vectors:,}</div>
    </div>
    <div class="metric-card">
        <div class="label">Index Name</div>
        <div class="value" style="font-size:1rem;">{index_name}</div>
    </div>
    <div class="metric-card">
        <div class="label">Embedding Model</div>
        <div class="value" style="font-size:0.75rem; font-family:'Inter',sans-serif;">all-MiniLM-L6-v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    top_k = st.slider("🎯 Chunks to retrieve (top_k)", min_value=1, max_value=10, value=3,
                      help="How many relevant passages to fetch from Pinecone per query")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="font-size:0.72rem; color:#b0a090; margin-top:1rem; text-align:center; line-height:1.6;">
        Secrets configured via<br><code>streamlit secrets.toml</code><br>
        GROQ_API_KEY · PINECONE_API_KEY<br>PINECONE_INDEX_NAME
    </div>
    """, unsafe_allow_html=True)


# ── Chat State ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Welcome Card (shown when no messages) ─────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>Begin Your Inquiry</h3>
        <p>Ask anything rooted in the divine teachings. The wisdom of Bhagavan Sri Sathya Sai Baba awaits your question.</p>
        <div class="sample-questions">
            <div class="sample-q">💬 "What does Swami say about love and service?"</div>
            <div class="sample-q">💬 "How can I control my mind and find inner peace?"</div>
            <div class="sample-q">💬 "What is the path to self-realization?"</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🪷"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips_html = "".join(
                f'<span class="source-chip">📄 {s}</span>' for s in msg["sources"]
            )
            st.markdown(f'<div class="source-chips">{chips_html}</div>', unsafe_allow_html=True)


# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Sai Baba's teachings…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🪷"):
        with st.spinner("Seeking wisdom…"):
            try:
                answer, sources = rag_chatbot(
                    prompt, groq_client, pinecone_index, embedding_model, top_k=top_k
                )
                st.markdown(answer)
                if sources:
                    chips_html = "".join(
                        f'<span class="source-chip">📄 {s}</span>' for s in sources
                    )
                    st.markdown(f'<div class="source-chips">{chips_html}</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
