import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from azure.storage.blob import BlobServiceClient
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import openai

# === CONFIG ===
st.set_page_config(page_title="MyMediTruth", layout="wide")
st.title("🩺 MyMediTruth: Unmasking Misinformation in Health-Care")

account_name = st.secrets["AZURE_STORAGE_ACCOUNT_NAME"]
container_name = st.secrets["AZURE_CONTAINER_NAME"]
sas_token = st.secrets["AZURE_SAS_TOKEN"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
blob_url = f"https://{account_name}.blob.core.windows.net"

# === UTILITY ===
def download_file(blob_path, local_path):
    if os.path.exists(local_path): return
    blob_service = BlobServiceClient(account_url=blob_url, credential=sas_token)
    blob = blob_service.get_container_client(container_name).get_blob_client(blob_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(blob.download_blob().readall())

@st.cache_data
def load_data():
    path = "data/mmt_merged.csv"
    download_file("data/mmt_merged.csv", path)
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)
    return df[df["label"] == "fake"]

@st.cache_resource
def load_model():
    model_path = "my_roberta_model"
    files = [
        "config.json", "model.safetensors", "tokenizer_config.json",
        "vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer.json"
    ]
    for f in files:
        download_file(f"my_roberta_model/{f}", f"{model_path}/{f}")
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

@st.cache_resource
def build_topic_model(tweets):
    vectorizer_model = CountVectorizer(stop_words='english')
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(tweets, show_progress_bar=False)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=False)
    topics, _ = topic_model.fit_transform(tweets, embeddings)
    return topic_model, topics

@st.cache_data
def generate_wordcloud(topics):
    exclusions = {"story", "study"}
    cleaned = [word for word in topics if all(x.lower() not in word.lower() for x in exclusions)]
    topic_words = " ".join(cleaned)
    wc = WordCloud(width=800, height=300, background_color="white", collocations=False).generate(topic_words)
    return wc

# === GPT Logic with Label Correction ===
def explain_claim(text, predicted_label):
    prompt = f"""
A user submitted the following self-care claim:
\"{text}\"

Is this claim real or fake? Respond with 'Label: Real' or 'Label: Fake', then explain why.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical fact-checking assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        explanation = response.choices[0].message.content.strip()

        if explanation.lower().startswith("label: real"):
            gpt_label = "real"
            explanation = explanation[len("label: real"):].strip()
        elif explanation.lower().startswith("label: fake"):
            gpt_label = "fake"
            explanation = explanation[len("label: fake"):].strip()
        else:
            gpt_label = predicted_label

        final_label = gpt_label if gpt_label != predicted_label else predicted_label
        return final_label, explanation

    except Exception as e:
        return predicted_label, f"(Could not generate explanation: {e})"

# === LOAD ===
df = load_data()
clf = load_model()

# === TOPIC MODELING ===
topic_model, topic_assignments = build_topic_model(df["text"].astype(str).tolist())
df["topic"] = topic_assignments

def get_topic_name(topic_id):
    if topic_id == -1:
        return "Uncategorized"
    top_words = [w for w, _ in topic_model.get_topic(topic_id)[:3]]
    return ", ".join(top_words).title()

df["Topic"] = df["topic"].apply(get_topic_name)

# === COLUMN CLEANING ===
df.rename(columns={
    "text": "Tweet",
    "tweet_count": "Tweet Count",
    "retweet_count": "Retweets",
    "reply_count": "Replies",
    "total_engagement": "Total Engagement"
}, inplace=True)

# === Word Cloud + Topic Table ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("☁️ Trending Themes in Fake Health News")
    filtered_topics = df[df["Topic"] != "Uncategorized"]["Topic"].tolist()
    wc = generate_wordcloud(filtered_topics)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.subheader("📈 Topic-Wise Engagement in Fake Health News")
    grouped = df.groupby("Topic").agg({
        "Tweet": "count",
        "Total Engagement": "mean",
        "Retweets": "mean",
        "Replies": "mean"
    }).reset_index()
    grouped.rename(columns={
        "Tweet": "# of Tweets",
        "Total Engagement": "Avg. Engagement",
        "Retweets": "Avg. Retweets",
        "Replies": "Avg. Replies"
    }, inplace=True)
    grouped = grouped.sort_values("Avg. Engagement", ascending=False)
    grouped[["Avg. Engagement", "Avg. Retweets", "Avg. Replies"]] = grouped[["Avg. Engagement", "Avg. Retweets", "Avg. Replies"]].round(0).astype(int)

    uncategorized = grouped[grouped["Topic"] == "Uncategorized"]
    grouped = grouped[grouped["Topic"] != "Uncategorized"]
    if not uncategorized.empty:
        grouped = pd.concat([grouped, uncategorized], ignore_index=True)

    st.dataframe(grouped, use_container_width=True)

# === Tweet Filter by Topic ===
st.markdown("---")
st.subheader("🔎 Explore Top Fake Health News Tweets by Topic")

topic_options = sorted(df["Topic"].unique())
default_topic = "Aspirin, Cancer, Death" if "Aspirin, Cancer, Death" in topic_options else topic_options[0]
selected_topic = st.selectbox("Choose a topic:", topic_options, index=topic_options.index(default_topic))
top_n = st.slider("Number of tweets to view:", min_value=3, max_value=30, value=5)

filtered = df[df["Topic"] == selected_topic].sort_values("Total Engagement", ascending=False).head(top_n)
filtered = filtered.reset_index(drop=True)
filtered.index += 1

tweet_table = filtered[["Tweet", "Tweet Count", "Retweets", "Replies", "Total Engagement"]].copy()
st.dataframe(tweet_table, use_container_width=True)

# === Claim Checker ===
st.markdown("---")
st.subheader("✅ Check a Self-Care Claim")

user_claim = st.text_area("Enter a claim to verify:", placeholder="e.g., Drinking lemon water detoxifies your liver.")

if st.button("Check Claim") and user_claim.strip():
    with st.spinner("Checking if the claim is health-related..."):
        domain_check_prompt = f"""
        Is the following statement related to health, medicine, or self-care? Only reply with 'Yes' or 'No'.

        Claim: \"{user_claim}\"
        """
        try:
            domain_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": domain_check_prompt.strip()}
                ],
                temperature=0,
                max_tokens=10
            )
            is_health = domain_response.choices[0].message.content.strip().lower().startswith("yes")
        except:
            is_health = True  # fallback to allow if GPT fails

    if not is_health:
        st.warning("⚠️ This doesn't appear to be a healthcare-related claim. Please enter a relevant statement.")
    else:
        with st.spinner("Analyzing..."):
            pred = clf([user_claim])[0]
            label = pred['label'].lower()
            final_label, explanation = explain_claim(user_claim, label)

        if final_label == "real":
            st.markdown("### 🟢 This claim appears to be **real**.")
        else:
            st.markdown("### 🔴 This claim appears to be **fake**.")

        st.markdown("#### Why this assessment?")
        st.write(explanation)
