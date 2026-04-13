# 🪷 Sai Teachings — RAG Chatbot

> A Retrieval-Augmented Generation chatbot built on the divine teachings of **Sri Sathya Sai Baba**, powered by Groq (LLaMA 3.1), Pinecone, and Streamlit.

---

## ✨ What It Does

Ask any question and receive answers grounded in Sathya Sai Baba's teachings — not from a generic AI, but from your own curated PDF library, semantically searched and retrieved in real time.

```
Your Question
     │
     ▼
Embed with all-MiniLM-L6-v2
     │
     ▼
Search Pinecone Vector Index  
     │
     ▼
Top-K Relevant Passages
     │
     ▼
Groq LLaMA 3.1 · Generate Answer
     │
     ▼
Response in the UI
```

---

## 🖥️ UI Preview

| Feature | Details |
|---|---|
| **Theme** | Clean white with warm gold accents |
| **Hero Header** | Animated lotus, title, connection status |
| **Chat** | Bubble-style, user right / assistant left |
| **Source Chips** | Shows which PDF file each answer came from |
| **Sidebar** | Index stats, `top_k` slider, clear chat |
| **Welcome Card** | Sample questions shown on first load |

---

## 🗂️ Project Structure

```
your-repo/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── secrets.toml.template   # Secrets reference (do NOT commit real keys)
└── README.md               # This file
```

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| **UI** | Streamlit |
| **LLM** | Groq API — `llama-3.1-8b-instant` |
| **Vector DB** | Pinecone (Serverless) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **PDF Parsing** | pypdf |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |

---

## 🚀 Deploy on Streamlit Share

### Step 1 — Push to GitHub

Make sure your repo has at minimum:

```
app.py
requirements.txt
```

> ⚠️ Do **not** commit `secrets.toml` or any file with real API keys.

---

### Step 2 — Add Secrets

Go to [share.streamlit.io](https://share.streamlit.io) → your app → **Settings → Secrets** and paste:

```toml
GROQ_API_KEY        = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_KEY    = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
PINECONE_INDEX_NAME = "xxxx"
```

| Secret | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `PINECONE_INDEX_NAME` | The name of your index, e.g. `saibooks` |

---

### Step 3 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create local secrets file
mkdir -p .streamlit
cp secrets.toml.template .streamlit/secrets.toml
# → Edit .streamlit/secrets.toml with your real keys

# 4. Run
streamlit run app.py
```

---



---

## 🔧 Configuration

### Adjusting `top_k`

Use the **sidebar slider** in the app to control how many passages are retrieved per query (default: 3, range: 1–10).

- Lower → faster, more focused answers
- Higher → broader context, potentially richer answers

### Changing the LLM Model

In `app.py`, find this line and swap the model name:

```python
model="llama-3.1-8b-instant",
```

Other Groq models you can use:

| Model | Speed | Quality |
|---|---|---|
| `llama-3.1-8b-instant` | ⚡ Fastest | Good |
| `llama-3.1-70b-versatile` | Moderate | Better |
| `mixtral-8x7b-32768` | Moderate | Good |

---

## 🛡️ Security Notes

- All API keys are stored in **Streamlit Secrets** — never in code
- The `.streamlit/secrets.toml` file is for local use only — add it to `.gitignore`
- The app is **read-only** — it only queries Pinecone, never modifies the index

```gitignore
# Add to your .gitignore
.streamlit/secrets.toml
```

---

## 🙏 Acknowledgements

Built with devotion to the teachings of **Bhagavan Sri Sathya Sai Baba**.

> *"Love all, Serve all. Help ever, Hurt never."*
> — Sri Sathya Sai Baba

---

## 📄 License

MIT — free to use, modify, and share.
