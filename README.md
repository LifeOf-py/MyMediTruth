# 🩺 MyMediTruth: Unmasking Misinformation in Health-Care

---

## 📖 Project Overview

**MyMediTruth** is an AI-powered application designed to detect and explain misleading or fake healthcare claims.  
We combine a fine-tuned **RoBERTa model** for classification with **GPT-4** explanations, deployed through a scalable **Streamlit** web app integrated with **Azure Blob Storage**.

**Key Highlights:**
- Classification of health-related claims into Real or Fake using fine-tuned RoBERTa
- Natural language explanations for claim validation using GPT-4
- Topic modeling and fake news trends analysis using BERTopic
- Fully cloud-enabled architecture (Azure, Streamlit Cloud)

---

> _This project repository is created in partial fulfillment of the requirements for the **Big Data Analytics** course offered by the **Master of Science in Business Analytics (MSBA)** program at the **Carlson School of Management, University of Minnesota**._

---

## 🛠 Setup and Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mymeditruth.git
cd mymeditruth
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Configure Streamlit Secrets
Create a .streamlit/secrets.toml file in the root directory with the following format:

```bash
AZURE_STORAGE_ACCOUNT_NAME = "your_account_name"
AZURE_CONTAINER_NAME = "your_container_name"
AZURE_SAS_TOKEN = "your_sas_token"
OPENAI_API_KEY = "your_openai_api_key"
```
**Important:** Never upload API keys or tokens to a public repository.

### 4. Run the App Locally
```bash
streamlit run app.py
```

### 5. Deployment (Optional)
You can deploy the app on Streamlit Cloud by linking your GitHub repository and configuring your Streamlit Secrets in the cloud workspace.

## Project Materials
[Streamlit App Link](https://mymeditruth.streamlit.app/)
[Project Flier](https://github.com/LifeOf-py/MyMediTruth/blob/main/MyMediTruth_Flyer_Team6.pdf)
[FakeHealth Dataset Reference](https://github.com/EnyanDai/FakeHealth)

## Technologies Used
- RoBERTa Fine-Tuning (HuggingFace Transformers)
- GPT-4 via OpenAI API
- BERTopic (Topic Modeling)
- Azure Blob Storage
- Streamlit App Framework
- Python (PyTorch, Scikit-learn, Pandas)

## Business Value
MyMediTruth enables faster identification and verification of misinformation in self-care and wellness spaces.
It is targeted toward:
- General users seeking fact-checked health advice
- Healthcare professionals validating viral claims
- Public health organizations tracking misinformation trends

## Acknowledgements
- [https://github.com/EnyanDai](https://github.com/EnyanDai/FakeHealth) for providing labeled health stories and engagement metrics.
- OpenAI, Azure, HuggingFace, and BERTopic open-source communities.
