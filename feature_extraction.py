"""
Sentinel URL - Phase 2
Step 1D - Feature Extraction
Run: python feature_extraction.py
Input : encoded_dataset.csv
Output: features_dataset.csv
"""

import pandas as pd
import numpy as np
import re
import math
import time

# ─────────────────────────────────────────
# FEATURE FUNCTIONS
# ─────────────────────────────────────────

def get_url_length(url):
    return len(url)

def get_dot_count(url):
    return url.count('.')

def get_hyphen_count(url):
    return url.count('-')

def get_slash_count(url):
    return url.count('/')

def get_at_count(url):
    return url.count('@')

def get_digit_count(url):
    return sum(c.isdigit() for c in url)

def get_special_char_count(url):
    return sum(1 for c in url if c in '!@#$%^&*()+=[]{}|;:,<>?')

def get_underscore_count(url):
    return url.count('_')

def get_question_count(url):
    return url.count('?')

def get_equal_count(url):
    return url.count('=')

def get_percent_count(url):
    return url.count('%')

def has_ip_address(url):
    pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(pattern, url) else 0

def has_https(url):
    return 1 if url.startswith('https') else 0

def has_http(url):
    return 1 if url.startswith('http://') else 0

def has_double_slash(url):
    return 1 if '//' in url[7:] else 0

def has_at_symbol(url):
    return 1 if '@' in url else 0

def get_domain_entropy(url):
    try:
        domain = url.split('/')[2] if '/' in url else url
        if len(domain) == 0:
            return 0
        prob = [domain.count(c) / len(domain) for c in set(domain)]
        entropy = -sum(p * math.log2(p) for p in prob if p > 0)
        return round(entropy, 4)
    except:
        return 0

def get_url_depth(url):
    try:
        path = url.split('/')
        return max(len(path) - 3, 0)
    except:
        return 0

def has_suspicious_keyword(url):
    keywords = [
        'login', 'signin', 'verify', 'secure', 'update',
        'bank', 'account', 'confirm', 'password', 'free',
        'lucky', 'prize', 'winner', 'click', 'urgent',
        'suspended', 'limited', 'unusual', 'authorize'
    ]
    return 1 if any(k in url.lower() for k in keywords) else 0

def has_suspicious_tld(url):
    tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq',
            '.top', '.vip', '.club', '.work', '.link']
    return 1 if any(url.lower().endswith(t) for t in tlds) else 0

def get_subdomain_count(url):
    try:
        domain = url.split('/')[2] if '/' in url else url
        parts = domain.split('.')
        return max(len(parts) - 2, 0)
    except:
        return 0

def has_port(url):
    try:
        domain = url.split('/')[2] if '/' in url else url
        return 1 if ':' in domain else 0
    except:
        return 0

def get_path_length(url):
    try:
        parts = url.split('/', 3)
        return len(parts[3]) if len(parts) > 3 else 0
    except:
        return 0

def has_encoded_chars(url):
    return 1 if '%' in url else 0

def get_digit_letter_ratio(url):
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    if letters == 0:
        return 0
    return round(digits / letters, 4)

# ─────────────────────────────────────────
# MAIN FEATURE EXTRACTOR
# ─────────────────────────────────────────

def extract_features(url):
    url = str(url)
    return {
        'url_length'          : get_url_length(url),
        'dot_count'           : get_dot_count(url),
        'hyphen_count'        : get_hyphen_count(url),
        'slash_count'         : get_slash_count(url),
        'at_count'            : get_at_count(url),
        'digit_count'         : get_digit_count(url),
        'special_char_count'  : get_special_char_count(url),
        'underscore_count'    : get_underscore_count(url),
        'question_count'      : get_question_count(url),
        'equal_count'         : get_equal_count(url),
        'percent_count'       : get_percent_count(url),
        'has_ip'              : has_ip_address(url),
        'has_https'           : has_https(url),
        'has_http'            : has_http(url),
        'has_double_slash'    : has_double_slash(url),
        'has_at_symbol'       : has_at_symbol(url),
        'domain_entropy'      : get_domain_entropy(url),
        'url_depth'           : get_url_depth(url),
        'suspicious_keyword'  : has_suspicious_keyword(url),
        'suspicious_tld'      : has_suspicious_tld(url),
        'subdomain_count'     : get_subdomain_count(url),
        'has_port'            : has_port(url),
        'path_length'         : get_path_length(url),
        'has_encoded_chars'   : has_encoded_chars(url),
        'digit_letter_ratio'  : get_digit_letter_ratio(url),
    }

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("="*55)
    print("  SENTINEL URL - Feature Extraction")
    print("="*55)

    # Load encoded dataset
    print("\n[STEP 1] Loading encoded_dataset.csv ...")
    df = pd.read_csv("encoded_dataset.csv")
    print(f"         Loaded {len(df):,} URLs")

    # Show class distribution
    print(f"\n[STEP 2] Class distribution:")
    label_names = {0:'benign', 1:'phishing', 2:'defacement', 3:'malware'}
    for lbl, name in label_names.items():
        cnt = len(df[df['label'] == lbl])
        print(f"         {lbl} = {name:<12} → {cnt:,} URLs")

    # Extract features
    print(f"\n[STEP 3] Extracting 25 features from each URL ...")
    print(f"         This may take 2-4 minutes for 640,003 URLs...")
    print(f"         Features: length, dots, entropy, keywords, TLD, IP, depth ...")

    start = time.time()
    features_list = []

    for i, url in enumerate(df['url']):
        features_list.append(extract_features(url))

        # Progress every 50,000 rows
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - start
            pct = (i + 1) / len(df) * 100
            print(f"         Progress: {i+1:,} / {len(df):,} ({pct:.1f}%) — {elapsed:.0f}s elapsed")

    elapsed = time.time() - start
    print(f"\n         Done! Extracted features in {elapsed:.1f} seconds ✅")

    # Build feature dataframe
    features_df = pd.DataFrame(features_list)

    # Add label column
    features_df['label'] = df['label'].values

    # Show feature summary
    print(f"\n[STEP 4] Feature matrix shape: {features_df.shape}")
    print(f"         Rows    : {features_df.shape[0]:,}  (one per URL)")
    print(f"         Columns : {features_df.shape[1]}  (25 features + 1 label)")

    print(f"\n[STEP 5] Sample feature values:")
    print(f"\n  {'Feature':<22} {'Benign':>10} {'Phishing':>10} {'Malware':>10} {'Defacement':>12}")
    print(f"  {'─'*66}")

    for feat in ['url_length','dot_count','domain_entropy',
                 'suspicious_keyword','has_ip','has_https','subdomain_count']:
        b = features_df[features_df['label']==0][feat].mean()
        p = features_df[features_df['label']==1][feat].mean()
        m = features_df[features_df['label']==3][feat].mean()
        d = features_df[features_df['label']==2][feat].mean()
        print(f"  {feat:<22} {b:>10.2f} {p:>10.2f} {m:>10.2f} {d:>12.2f}")

    # Save
    print(f"\n[STEP 6] Saving features_dataset.csv ...")
    features_df.to_csv("features_dataset.csv", index=False)
    print(f"         Saved! ({len(features_df):,} rows × {features_df.shape[1]} columns)")

    print(f"\n{'='*55}")
    print(f"  FEATURE EXTRACTION COMPLETE! ✅")
    print(f"{'='*55}")
    print(f"  Input  : encoded_dataset.csv")
    print(f"  Output : features_dataset.csv")
    print(f"  Features extracted : 25")
    print(f"  Total URLs         : {len(features_df):,}")
    print(f"\n  Next step → run step1_model_comparison.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
