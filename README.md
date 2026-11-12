# SentinelURL: AI Framework for Predictive Phishing Intelligence

An advanced, real-time machine learning threat classification framework engineered to detect evolving multi-class phishing and malware URLs that bypass traditional static blacklists.

## 🚀 Overview
Traditional blacklists struggle to keep pace with zero-day phishing attacks and sophisticated social engineering tactics. **SentinelURL** leverages lexical feature extraction combined with a high-performance Random Forest classifier to evaluate URL security instantly.

Developed as a Major Project (Phase 2 - BCS786) at the **Don Bosco Institute of Technology, Bengaluru**.

---

## 📊 Key Performance Metrics
Evaluated on a robust dataset of over 640,000+ labeled URLs (Benign, Phishing, Defacement, and Malware):

* **Accuracy**: 94.28%
* **ROC-AUC**: 0.9851
* **Classifier**: Random Forest with full cross-validation
* **Features**: 12+ engineered lexical and network metrics (Shannon Entropy, URL length, special character density, and WHOIS OSINT signals)

---

## 🛠️ System Architecture & Data Flow
1. **URL Input**: Raw URL ingestion.
2. **Feature Preprocessing**: Extraction of lexical attributes and network/WHOIS metadata.
3. **Model Prediction**: Evaluation via optimized Random Forest classifier.
4. **Output**: Real-time risk score and threat classification dashboard.

---

## 📁 Repository Structure
```text
├── sentinel_extension/     # Browser extension client interface (Manifest v3)
├── app.py                  # Main application backend
├── feature_extraction.py   # Lexical and WHOIS feature parsing pipelines
├── clean_and_encode.py     # Dataset preprocessing and encoding scripts
├── step1_model_comparison.py# Model benchmarking (Random Forest, XGBoost, Logistic Regression)
├── step2b_charts.py        # Performance visualization script
├── requirements.txt        # Python dependencies
└── *.png                   # Evaluation charts (Confusion matrix, ROC curves, feature importance)

(Note: Trained model binaries like best_model.pkl and heavy CSV datasets are excluded from this repository due to GitHub's 100 MB file size limits).


⚙️ Installation & Setup
Clone the Repository

Bash
git clone [https://github.com/Roshini-s555/Sentinel-url-phishing-detection.git](https://github.com/Roshini-s555/Sentinel-url-phishing-detection.git)
cd Sentinel-url-phishing-detection
Install Dependencies

Bash
pip install -r requirements.txt
Run the Application Locally

Bash
python app.py
👥 Authors
Roshini S. (Department of Computer Science & Engineering, Batch 2027)


<ElicitationsGroup message="Once you've added the README file, what would you like to do next?">
  <Elicitation label="Commit and push README" query="What are the exact git commands to add, commit, and push this new README.md file to GitHub?"/>
  <Elicitation label="Test local run" query="How do I run app.py locally to test the application?"/>
  <Elicitation label="Prepare presentation slides" query="How can I prepare for my major project presentation using the generated evaluation charts?"/>
</ElicitationsGroup>
