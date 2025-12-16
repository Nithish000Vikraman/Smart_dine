import streamlit as st
import pandas as pd
import json, os, hashlib, urllib.parse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config("SmartDine 🍽️", "🍕", layout="wide")
DATA_FILE = "users_data.json"

# -------------------------------------------------
# SECURITY
# -------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------------------------------
# USER STORAGE
# -------------------------------------------------
def load_users():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def maps_link(location):
    return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(location)}"

def cuisine_image(cuisine):
    return f"https://source.unsplash.com/800x500/?{cuisine.replace(' ', ',')}"

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "users" not in st.session_state:
    st.session_state.users = load_users()

if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot_messages" not in st.session_state:
    st.session_state.bot_messages = []

# -------------------------------------------------
# AUTH
# -------------------------------------------------
if not st.session_state.user:
    st.markdown("## 🔐 SmartDine Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in st.session_state.users and \
               st.session_state.users[u]["password"] == hash_password(p):
                st.session_state.user = u
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")
        if st.button("Register"):
            if nu and nu not in st.session_state.users:
                st.session_state.users[nu] = {
                    "password": hash_password(np),
                    "saved": []
                }
                save_users(st.session_state.users)
                st.success("Account created. Login now.")
            else:
                st.error("Username already exists")

    st.stop()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=120
    )

embed_model = load_embed_model()
llm = load_llm()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/restaurants.csv")

df = load_data()

# -------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------
@st.cache_data
def prepare_embeddings(data):
    text = (
        data["name"] + " " +
        data["cuisine"] + " " +
        data["price_range"] + " " +
        data["description"]
    ).tolist()
    return embed_model.encode(text)

embeddings = prepare_embeddings(df)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.markdown(f"👋 **{st.session_state.user}**")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.experimental_rerun()

budget = st.sidebar.selectbox("💰 Budget", ["Any", "Low", "Medium", "High"])
surprise = st.sidebar.button("🎲 Surprise Me")

# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown("""
<div style="background:linear-gradient(120deg,#ff512f,#dd2476);
padding:40px;border-radius:28px;text-align:center;color:white;">
<h1>🍽️ SmartDine</h1>
<p>AI Food Discovery + LLM Chatbot</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_chat, tab_saved, tab_bot = st.tabs(
    ["💬 Food Chat", "❤️ Saved", "🤖 AI Chatbot"]
)

# -------------------------------------------------
# RECOMMENDER
# -------------------------------------------------
def recommend(query):
    q = embed_model.encode([query])
    scores = cosine_similarity(q, embeddings)[0]
    data = df.copy()
    data["score"] = scores
    if budget != "Any":
        data = data[data["price_range"] == budget]
    return data.sort_values("score", ascending=False).head(4)

# -------------------------------------------------
# FOOD CHAT TAB
# -------------------------------------------------
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    query = st.chat_input("What are you craving today?")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        results = recommend(query)

        with st.chat_message("assistant"):
            st.markdown("Here are some great picks 👇")

        for idx, r in results.iterrows():
            st.image(cuisine_image(r["cuisine"]), width=700)

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.08);
            padding:18px;border-radius:18px;">
            <h3>{r['name']}</h3>
            🍴 {r['cuisine']} | 💰 {r['price_range']} | ⭐ {r['rating']}<br>
            📍 {r['location']}<br><br>
            <i>{r['description']}</i>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            if col1.button("❤️ Save", key=f"s_{idx}"):
                saved = st.session_state.users[st.session_state.user]["saved"]
                if r["name"] not in saved:
                    saved.append(r["name"])
                    save_users(st.session_state.users)
                    st.success("Saved!")

            col2.markdown(f"[🗺️ Open in Maps]({maps_link(r['location'])})")

# -------------------------------------------------
# SAVED TAB
# -------------------------------------------------
with tab_saved:
    saved = st.session_state.users[st.session_state.user]["saved"]
    search = st.text_input("🔍 Search saved")

    for r in [x for x in saved if search.lower() in x.lower()]:
        loc = df[df["name"] == r]["location"].values[0]
        st.markdown(f"🍽️ **{r}** | [🗺️ Maps]({maps_link(loc)})")

# -------------------------------------------------
# 🤖 LLM CHATBOT TAB
# -------------------------------------------------
with tab_bot:
    st.markdown("## 🤖 SmartDine AI Assistant")
    st.caption("Ask anything: food tips, diets, tech, life advice")

    for m in st.session_state.bot_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask me anything...")

    if user_q:
        st.session_state.bot_messages.append(
            {"role": "user", "content": user_q}
        )
        with st.chat_message("user"):
            st.markdown(user_q)

        prompt = "Conversation:\n"
        for m in st.session_state.bot_messages[-6:]:
            prompt += f"{m['role']}: {m['content']}\n"
        prompt += "assistant:"

        response = llm(prompt)[0]["generated_text"].split("assistant:")[-1]

        st.session_state.bot_messages.append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.markdown(response)

# -------------------------------------------------
# SURPRISE
# -------------------------------------------------
if surprise:
    pick = df.sample(1).iloc[0]
    st.toast(f"🎉 Try {pick['name']} today!", icon="🍕")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
SmartDine – AI Food Discovery + LLM Chatbot
</p>
""", unsafe_allow_html=True)
