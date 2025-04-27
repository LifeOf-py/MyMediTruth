#############################
# DATA PREP
#############################

# First get the data folder(I downloaded as zip in local and then extracted it)
# from the github repo and store it somewhere - you can follow the path 
# I have used to check how it is supposed to be  

import json
import pandas as pd

# === Load the JSON file ===
review_path = "FakeHealth-master/dataset/reviews/HealthStory.json"

with open(review_path, "r", encoding="utf-8") as f:
    review_data = json.load(f)

review_records = []

for item in review_data:
    story_id = item.get("news_id", "").strip()
    if not story_id:
        continue

    # === Count "Satisfactory" and "Not Satisfactory" ===
    not_sat = 0
    sat = 0
    criteria = item.get("criteria", [])
    for criterion in criteria:
        answer = criterion.get("answer", "").strip().lower()
        if answer == "not satisfactory":
            not_sat += 1
        elif answer == "satisfactory":
            sat += 1

    # === Assign Binary Label ===
    if sat >= 7 and not_sat <= 3:
        label = "real"
    else:
        label = "fake"  # includes fake + misleading

    # === Extract summary text safely ===
    summary_field = item.get("summary", "")
    if isinstance(summary_field, dict):
        summary_text = summary_field.get("Our Review Summary", "")
    else:
        summary_text = summary_field

    # === Compose input text ===
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        summary_text,
        item.get("Why This Matters", "")
    ]

    full_text = "\n\n".join([p.strip() for p in parts if isinstance(p, str) and p.strip()])

    if full_text:
        review_records.append({
            "story_id": story_id,
            "text": full_text,
            "label": label
        })

# === Create Final DataFrame ===
df_reviews = pd.DataFrame(review_records)

# === Show result ===
df_reviews.to_csv("mymeditruth_bert_dataset.csv", index = False)
print(f"Total processed: {len(df_reviews)}")
print(df_reviews['label'].value_counts())

######
###### CREATE THIS AND STORE IN A CSV FOR NOW - NOT NEEDED FOR TRAINING AS OF NOW
######
# Load the consolidated JSON file
engagement_path = "FakeHealth-master/dataset/engagements/HealthStory.json"

with open(engagement_path, "r", encoding="utf-8") as f:
    engagement_data = json.load(f)

# Extract engagement summary per story
engagement_records = []

for story_id, content in engagement_data.items():
    tweets = content.get("tweets", [])
    retweets = content.get("retweets", [])
    replies = content.get("replies", [])

    engagement_records.append({
        "story_id": story_id,
        "tweet_count": len(tweets),
        "retweet_count": len(retweets),
        "reply_count": len(replies),
        "total_engagement": len(tweets) + len(retweets) + len(replies)
    })

# Convert to DataFrame
df_engagement = pd.DataFrame(engagement_records)

# --- Join Reviews + Engagements ---
df_merged = df_reviews.merge(df_engagement, on="story_id", how="left")

# Fill missing engagement with 0s
df_merged.fillna({
    "tweet_count": 0, "retweet_count": 0, "reply_count": 0, "total_engagement": 0}, inplace=True)



#############################
# MODEL TRAINING - WE ARE TRAINING ON WHOLE DATA
#############################
import pandas as pd
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoConfig,
    Trainer, TrainingArguments
)
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss

# ===== 1. Load and Prepare Full Data =====
df_reviews = pd.read_csv("mymeditruth_bert_dataset.csv")
df_reviews = df_reviews[df_reviews["label"].isin(["fake", "real"])]
df_reviews["label_id"] = df_reviews["label"].map({"fake": 0, "real": 1})
full_df = df_reviews.copy()

# ===== 2. Tokenization =====
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
encodings = tokenizer(full_df["text"].tolist(), truncation=True, padding=True, max_length=512)

# ===== 3. Dataset Class =====
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

dataset = ReviewDataset(encodings, full_df["label_id"].tolist())

# ===== 4. Compute Class Weights =====
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(full_df["label_id"]),
    y=full_df["label_id"]
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ===== 5. Custom Weighted Loss Model =====
class WeightedLossRoberta(RobertaForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

# ===== 6. Load Config & Model =====
config = AutoConfig.from_pretrained("roberta-base", num_labels=2)
model = WeightedLossRoberta.from_pretrained("roberta-base", config=config, class_weights=class_weights_tensor)

# ===== 7. Training Arguments =====
training_args = TrainingArguments(
    output_dir="./mymeditruth_roberta_final",
    save_strategy="epoch",
    logging_dir="./logs_roberta_final",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    report_to="none"
)

# ===== 8. Train on Full Dataset =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./mymeditruth_roberta_final")
tokenizer.save_pretrained("./mymeditruth_roberta_final")

print("✅ Training complete. Final model saved to './mymeditruth_roberta_final'")


#############################
# MODEL TESTING
#############################
# ✅ Step 1: Define sample claims
# claims = [
#     "Rubbing garlic on your feet before bed helps you detox through your soles overnight.",
#     "Putting ice cubes on your neck every morning resets your nervous system and boosts mental clarity.",
#     "Using niacinamide helps reduce dark spots on the face.",
#     "Drinking chlorophyll water daily will completely clear your skin."
# ]

claims = [
    "Daily sunlight reduces chanced of getting prostate cancer"
]

# ✅ Step 2: Load fine-tuned classification model
from transformers import pipeline

model_path = "./mymeditruth_roberta_final"
clf_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path)

# ✅ Step 3: Run predictions
results = clf_pipeline(claims)

# ✅ Step 4: GPT-based explanation and fallback labeling (OpenAI SDK v1.x)
import openai
client = openai.OpenAI(api_key="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")  # Replace securely

def explain_claim(text, predicted_label, confidence, threshold=0.85):
    prompt = f"""
A user submitted the following self-care claim:

"{text}"

Is this claim real or fake? Respond with 'Label: Real' or 'Label: Fake', then provide a medically grounded explanation in 3–5 sentences.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical fact-checking assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        explanation = response.choices[0].message.content.strip()

        # Parse label and strip it from explanation
        if explanation.lower().startswith("label: fake"):
            gpt_label = "Fake"
            explanation = explanation[len("label: fake"):].strip()
        elif explanation.lower().startswith("label: real"):
            gpt_label = "Real"
            explanation = explanation[len("label: real"):].strip()
        else:
            gpt_label = "Unclear"

        # Decision logic: if GPT disagrees with model OR model confidence < threshold, use GPT's label
        if confidence < threshold or gpt_label.lower() != predicted_label:
            label = gpt_label
        else:
            label = predicted_label.capitalize()  # match casing

        return label, explanation

    except Exception as e:
        return "Error", f"Error: {str(e)}"

# ✅ Step 5: Combine prediction + explanation + GPT fallback
for claim, prediction in zip(claims, results):
    predicted_label = prediction['label'].lower()
    confidence = prediction['score']

    final_label, explanation = explain_claim(claim, predicted_label, confidence)

    print(f"\n🧾 Claim: {claim}")
    print(f"📊 Final Label: {final_label} (Model confidence: {confidence:.2f})")
    print(f"💬 Explanation: {explanation}")
    print("-" * 80)

