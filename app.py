from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
import math
import whois
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS so your frontend and Chrome extension can talk to it

# Load the saved model and scaler
print("Loading model and scaler...")
model_data = joblib.load("best_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
print(f"Loaded model: {model_data['model_name']}")

# List of guaranteed safe domains (Whitelist to prevent false positives on famous sites)
WHITELISTED_DOMAINS = [
    "google.com", "microsoft.com", "github.com", "wikipedia.org",
    "amazon.in", "amazon.com", "youtube.com", "netflix.com", "apple.com",
    "stackoverflow.com", "python.org"
]

# Feature extraction function (same as used in training)
def extract_features(url):
    url = str(url)
    features = {}
    features["url_length"]        = len(url)
    features["dot_count"]         = url.count(".")
    features["hyphen_count"]      = url.count("-")
    features["slash_count"]       = url.count("/")
    features["at_count"]          = url.count("@")
    features["question_count"]    = url.count("?")
    features["equal_count"]       = url.count("=")
    features["underscore_count"]  = url.count("_")
    features["percent_count"]     = url.count("%")
    features["digit_count"]       = sum(c.isdigit() for c in url)
    features["has_ip"]            = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0
    features["has_https"]         = 1 if url.startswith("https") else 0
    features["has_http"]          = 1 if url.startswith("http://") else 0
    features["has_double_slash"]  = 1 if "//" in url[7:] else 0
    features["has_at_symbol"]     = 1 if "@" in url else 0
    try:
        domain = url.split("/")[2] if "/" in url else url
        prob   = [domain.count(c) / len(domain) for c in set(domain)]
        entropy = -sum(p * math.log2(p) for p in prob if p > 0)
    except:
        entropy = 0
    features["domain_entropy"]    = round(entropy, 4)
    features["url_depth"]         = url.count("/") - 2 if url.count("/") > 2 else 0
    suspicious_words = ["login","signin","verify","secure","update",
                        "bank","account","confirm","password","free"]
    features["suspicious_keyword"] = 1 if any(w in url.lower() for w in suspicious_words) else 0
    suspicious_tlds = [".xyz",".tk",".ml",".ga",".cf",".gq",".top",".vip"]
    features["suspicious_tld"]    = 1 if any(url.lower().endswith(t) for t in suspicious_tlds) else 0
    return features

# Domain Age Lookup via WHOIS
def get_domain_age(url):
    try:
        domain = url.split("/")[2] if "/" in url else url
        domain = domain.split(":")[0]  # Strip port if any
        
        w = whois.whois(domain)
        creation_date = w.creation_date
        
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
            
        if not creation_date:
            return "Unknown"
            
        age_days = (datetime.now() - creation_date).days
        return f"{age_days} days old"
    except Exception as e:
        return "Unavailable / Protected"

CLASS_MAPPING = {
    0: {"name": "Benign", "risk": "Safe"},
    1: {"name": "Phishing", "risk": "High"},
    2: {"name": "Defacement", "risk": "Medium"},
    3: {"name": "Malware", "risk": "Critical"}
}

@app.route("/")
def home():
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url = data.get("url", "")
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # 1. Check Whitelist First to prevent false positives on major sites
    for safe_domain in WHITELISTED_DOMAINS:
        if safe_domain in url.lower():
            return jsonify({
                "url": url,
                "prediction": "Benign",
                "risk_level": "Safe",
                "domain_age": "Verified Enterprise Domain",
                "probabilities": {
                    "Benign": 100.0,
                    "Phishing": 0.0,
                    "Defacement": 0.0,
                    "Malware": 0.0
                }
            })
    
    # 2. Extract features
    feats = extract_features(url)
    X = pd.DataFrame([feats])
    
    # 3. Scale features
    X_scaled = scaler.transform(X)
    
    # 4. Predict
    pred_label = int(model.predict(X_scaled)[0])
    pred_proba = model.predict_proba(X_scaled)[0].tolist()
    
    # 5. Get Domain Age OSINT info
    domain_age = get_domain_age(url)
    
    class_info = CLASS_MAPPING.get(pred_label, {"name": "Unknown", "risk": "Unknown"})
    
    response = {
        "url": url,
        "prediction": class_info["name"],
        "risk_level": class_info["risk"],
        "domain_age": domain_age,
        "probabilities": {
            "Benign": round(pred_proba[0] * 100, 2),
            "Phishing": round(pred_proba[1] * 100, 2),
            "Defacement": round(pred_proba[2] * 100, 2),
            "Malware": round(pred_proba[3] * 100, 2)
        }
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)