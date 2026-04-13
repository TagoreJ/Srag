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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero ── */
.hero {
    padding: 2.2rem 2rem 1.6rem;
    text-align: center;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 0.5rem;
}
.hero-lotus {
    font-size: 2.6rem;
    display: block;
    margin-bottom: 0.4rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 600;
    margin: 0 0 0.3rem;
    color: inherit;
}
.hero-sub {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.55;
    margin: 0;
    color: inherit;
}
.hero-rule {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, #c9956a, #d4af6b);
    border-radius: 2px;
    margin: 0.9rem auto 0;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 1rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.35);
    color: #16a34a;
    margin: 0.8rem auto 0.4rem;
}
.status-dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(1.4); }
}

/* ── Index badge ── */
.index-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.9rem;
    border-radius: 8px;
    font-size: 0.76rem;
    font-weight: 500;
    background: rgba(212,175,107,0.15);
    border: 1px solid rgba(212,175,107,0.35);
    color: #92651a;
    margin-top: 0.3rem;
}

/* ── Welcome card ── */
.welcome-card {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    text-align: center;
    margin: 1.2rem auto 1.6rem;
    max-width: 540px;
    background: rgba(212,175,107,0.07);
    border: 1px solid rgba(212,175,107,0.2);
}
.welcome-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0 0 0.5rem;
    color: inherit;
}
.welcome-card p {
    font-size: 0.86rem;
    line-height: 1.7;
    opacity: 0.7;
    margin: 0 0 1rem;
    color: inherit;
}
.sample-q {
    display: block;
    text-align: left;
    padding: 0.5rem 0.85rem;
    margin-bottom: 0.4rem;
    border-radius: 10px;
    font-size: 0.82rem;
    font-style: italic;
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.15);
    color: inherit;
    opacity: 0.8;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    background: linear-gradient(135deg, #b07d2a, #c9956a) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    font-size: 0.94rem;
    line-height: 1.65;
    box-shadow: 0 2px 10px rgba(180,120,40,0.2);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
    background: rgba(128,128,128,0.09) !important;
    border: 1px solid rgba(128,128,128,0.18) !important;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.1rem;
    font-size: 0.94rem;
    line-height: 1.7;
    color: inherit !important;
}

/* ── Source chips ── */
.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.45rem;
}
.chip {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 20px;
    font-size: 0.71rem;
    font-weight: 500;
    background: rgba(212,175,107,0.15);
    border: 1px solid rgba(212,175,107,0.3);
    color: #92651a;
}

/* ── Sidebar stat cards ── */
.stat-card {
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    text-align: center;
    background: rgba(212,175,107,0.08);
    border: 1px solid rgba(212,175,107,0.2);
}
.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    color: #92651a;
    margin-bottom: 0.15rem;
}
.stat-value {
    font-size: 1.45rem;
    font-weight: 700;
    font-family: 'Playfair Display', serif;
    color: inherit;
}
.stat-value-sm {
    font-size: 0.82rem;
    font-weight: 500;
    color: inherit;
    opacity: 0.85;
    word-break: break-all;
}

/* ── About text ── */
.about-text {
    font-size: 0.83rem;
    line-height: 1.7;
    opacity: 0.75;
    color: inherit;
}

/* ── Error box ── */
.err-box {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.86rem;
    line-height: 1.65;
    background: rgba(220,38,38,0.08);
    border: 1px solid rgba(220,38,38,0.25);
    color: #dc2626;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.3); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Load clients ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_clients():
    groq_api_key     = st.secrets["GROQ_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    index_name       = st.secrets["PINECONE_INDEX_NAME"]

    pc    = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    groq  = Groq(api_key=groq_api_key)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    stats = index.describe_index_stats()
    return groq, index, model, index_name, stats


# ── RAG ───────────────────────────────────────────────────────────────────────
def rag_chatbot(user_query, groq_client, pinecone_index, embedding_model, top_k=3):
    query_embedding = embedding_model.encode(user_query).tolist()

    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    chunks, sources = [], []
    for match in results.matches:
        meta = match.get("metadata", {})
        if meta.get("text"):
            chunks.append(meta["text"])
        fname = meta.get("file_name", "")
        if fname and fname not in sources:
            sources.append(fname)

    if not chunks:
        return "I could not find relevant teachings to answer your question.", []

    system_msg = (
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
    user_msg = (
        f"Question:\n{user_query}\n\n"
        f"Relevant teachings:\n{chr(10).join(chunks[:2])}\n\n"
        "Answer briefly and gently."
    )

    resp = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        model="llama-3.1-8b-instant",
    )
    return resp.choices[0].message.content, sources


# ── Init ──────────────────────────────────────────────────────────────────────
try:
    groq_client, pinecone_index, embedding_model, index_name, stats = init_clients()
    total_vectors = stats.total_vector_count
    ready = True
except Exception as e:
    ready = False
    init_error = str(e)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-lotus">🪷</span>
    <h1>Sai Teachings</h1>
    <p class="hero-sub">Ask · Seek · Receive Wisdom</p>
    <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

if not ready:
    st.markdown(f"""
    <div class="err-box">
        ⚠️ <strong>Could not connect.</strong> Please check your Streamlit secrets.<br>
        Required: <code>GROQ_API_KEY</code> · <code>PINECONE_API_KEY</code> · <code>PINECONE_INDEX_NAME</code><br><br>
        Error: {init_error}
    </div>""", unsafe_allow_html=True)
    st.stop()

# Status badges
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown(f"""
    <div style="text-align:center;">
        <div class="status-pill">
            <span class="status-dot"></span>
            Connected · {total_vectors:,} teachings indexed
        </div><br>
        <div class="index-badge">📚 Index: <strong>{index_name}</strong></div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪷 About")
    st.markdown("""
    <p class="about-text">
        This chatbot draws wisdom from the teachings of <strong>Sri Sathya Sai Baba</strong>,
        stored in a Pinecone vector database and answered by <strong>Groq LLaMA 3.1</strong>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Index Stats**")
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Teachings</div>
        <div class="stat-value">{total_vectors:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Index Name</div>
        <div class="stat-value-sm">{index_name}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Embedding Model</div>
        <div class="stat-value-sm">all-MiniLM-L6-v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    top_k = st.slider(
        "🎯 Chunks to retrieve (top_k)",
        min_value=1, max_value=10, value=3,
        help="How many relevant passages to fetch from Pinecone per query",
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>Begin Your Inquiry</h3>
        <p>Ask anything rooted in the divine teachings. The wisdom of Bhagavan Sri Sathya Sai Baba awaits your question.</p>
        <span class="sample-q">💬 "What does Swami say about love and service?"</span>
        <span class="sample-q">💬 "How can I control my mind and find inner peace?"</span>
        <span class="sample-q">💬 "What is the path to self-realization?"</span>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🪷"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(f'<span class="chip">📄 {s}</span>' for s in msg["sources"])
            st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask about Sai Baba's teachings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🪷"):
        with st.spinner("Seeking wisdom…"):
            try:
                answer, sources = rag_chatbot(
                    prompt, groq_client, pinecone_index, embedding_model, top_k=top_k
                )
                st.markdown(answer)
                if sources:
                    chips = "".join(f'<span class="chip">📄 {s}</span>' for s in sources)
                    st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources,
                })
            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})