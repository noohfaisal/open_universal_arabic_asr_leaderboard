import json
import os
import re
import numpy as np

# Normalization (Copied from leaderboard_score.py for consistency)
def normalize_arabic_text(text):
    punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)
    diacritics = r'[\u064B-\u0652]'
    text = re.sub(diacritics, '', text)
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)   
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)
    return text.strip()

# Simple WER implementation
def calculate_wer(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()
    return _levenshtein(r, h)

def calculate_cer(reference, hypothesis):
    # For CER, we treat the string as a list of characters (excluding spaces usually, or including? 
    # Standard CER often includes spaces or treats them as chars. 
    # Let's simple list(text) which includes spaces if not stripped.
    # But usually normalization removes some. Let's use standard list(text.replace(" ", "")) for tight CER 
    # OR list(text) for standard Levenshtein on chars. 
    # Given the normalization function doesn't strip internal spaces, we stick to list(text).
    r = list(reference)
    h = list(hypothesis)
    
    # Re-use the matrix logic or generic distance
    # To keep it simple/fast for this script, we can reuse logic but it's redundant.
    # Let's just copy-paste for safety/independence or genericize.
    # Genericizing:
    return _levenshtein(r, h)

def _levenshtein(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.int32)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j]) + 1
    
    return d[len(r)][len(h)], len(r)



manifest_path = "results/whisper/sada_output.json"

total_w_dist = 0
total_w_len = 0
total_c_dist = 0
total_c_len = 0
individual_wers = []
individual_cers = []
lengths = []

print(f"Analyzing {manifest_path}...")

with open(manifest_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        ref = normalize_arabic_text(item['text'])
        hyp = normalize_arabic_text(item['pred_text'])
        
        # WER
        w_dist, w_len = calculate_wer(ref, hyp)
        wer = w_dist / w_len if w_len > 0 else (1.0 if len(hyp) > 0 else 0.0)
        
        # CER
        c_dist, c_len = calculate_cer(ref, hyp)
        cer = c_dist / c_len if c_len > 0 else (1.0 if len(hyp) > 0 else 0.0)
                
        individual_wers.append(wer)
        individual_cers.append(cer)
        lengths.append(w_len)
        
        total_w_dist += w_dist
        total_w_len += w_len
        total_c_dist += c_dist
        total_c_len += c_len

# Metrics
global_wer = total_w_dist / total_w_len if total_w_len > 0 else 0.0
mean_wer = sum(individual_wers) / len(individual_wers) if individual_wers else 0.0
global_cer = total_c_dist / total_c_len if total_c_len > 0 else 0.0
mean_cer = sum(individual_cers) / len(individual_cers) if individual_cers else 0.0

print("-" * 30)
print(f"Total Prediction Items: {len(lengths)}")
print("-" * 30)
print(f"Global WER (Matches Leaderboard): {global_wer:.4f}")
print(f"Mean WER   (Matches Dashboard):   {mean_wer:.4f}")
print("-" * 30)
print(f"Global CER (Matches Leaderboard): {global_cer:.4f}")
print(f"Mean CER   (Matches Dashboard):   {mean_cer:.4f}")
print("-" * 30)
print(f"Avg Reference Length (Words): {sum(lengths)/len(lengths):.2f}")

