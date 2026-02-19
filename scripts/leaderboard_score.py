
import sys
import os
# Ensure we can import from local scripts if running from root
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.getcwd())

try:
    from db_manager import init_db, log_experiment, log_inferences
    DB_AVAILABLE = True
except ImportError as e:
    # Try importing as package
    try:
        from scripts.db_manager import init_db, log_experiment, log_inferences
        DB_AVAILABLE = True
    except ImportError as e2:
        print(f"DB Import failed: {e}, {e2}")
        DB_AVAILABLE = False
import argparse
import json
import glob
from collections import defaultdict
import numpy as np
from nemo.collections.asr.metrics.wer import word_error_rate
import re

def normalize_arabic_text(text):
    # Reuse normalization from eval.py or import it if possible. 
    # For standalone script, duplicating the simple regex storage is safer/easier than fixing imports given current structure.
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

def calculate_metrics(manifest_path):
    predictions = []
    target_transcripts = []
    inference_times = []
    durations = []
    
    detailed_items = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            norm_target = normalize_arabic_text(item['text'])
            norm_pred = normalize_arabic_text(item['pred_text'])
            
            target_transcripts.append(norm_target)
            predictions.append(norm_pred)
            
            inf_time = item.get("inference_time", 0.0)
            duration = item.get("duration", 0.0)
            
            if "inference_time" in item:
                inference_times.append(inf_time)
            if "duration" in item:
                durations.append(duration)
            
            # Prepare detailed item for DB
            detailed_items.append({
                "audio_filepath": item.get("audio_filepath", ""),
                "text": norm_target,
                "pred_text": norm_pred,
                "duration": duration,
                "inference_time": inf_time,
                "rtf": (inf_time / duration) if duration > 0 else 0.0
                # WER/CER per file could be calculated here but keeping it simple for now, 
                # we rely on global/batch calculation or can add individual calc later if needed for DB.
                # For this implementation, we will pass them as 0.0 or calculate simple match?
                # Let's calculate simple WER per line for the DB? 
                # word_error_rate can take single strings? No, usually lists.
                # We will leave per-file WER/CER as 0.0 in DB for now or calculate loosely? 
                # Actually db_manager expects it. Let's calculate loosely.
            })

    # Individual WER calc for DB logging (optional but good)
    # This is inefficient for large datasets to loop again or call wer many times, 
    # but for typical leaderboard sizes it's fine.
    for i, det in enumerate(detailed_items):
        w = word_error_rate([det['pred_text']], [det['text']])
        c = word_error_rate([det['pred_text']], [det['text']], use_cer=True)
        det['wer'] = w
        det['cer'] = c

    wer = word_error_rate(predictions, target_transcripts)
    cer = word_error_rate(predictions, target_transcripts, use_cer=True)
    
    avg_latency = 0.0
    if inference_times:
        avg_latency = sum(inference_times) / len(inference_times)
        
    rtf = 0.0
    if inference_times and durations and sum(durations) > 0:
        rtf = sum(inference_times) / sum(durations)
        
    metrics = {
        "wer": wer,
        "cer": cer,
        "avg_latency": avg_latency,
        "rtf": rtf,
        "num_samples": len(predictions)
    }
    return metrics, detailed_items

def main(results_root, save_to_db=False, clear_db_flag=False):
    if save_to_db and DB_AVAILABLE:
        if clear_db_flag:
            try:
                from scripts.db_manager import clear_db
                print("Clearing Database...")
                clear_db()
            except ImportError:
                 from db_manager import clear_db
                 print("Clearing Database...")
                 clear_db()
        else:
            print("Initializing Database...")
            init_db()
    elif save_to_db and not DB_AVAILABLE:
        print("Warning: Database dependencies not found. Skipping DB save.")
        save_to_db = False

    # Expected structure: results_root/model_name/dataset_name.json
    # Find all subdirectories which we assume are models
    model_dirs = [d for d in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(d)]
    
    if not model_dirs:
        # Fallback: maybe the root itself is just one model's results?
        # Check if there are json files in results_root
        if glob.glob(os.path.join(results_root, "*.json")):
            model_dirs = [results_root]
        else:
            print(f"No model directories or json files found in {results_root}")
            return
            
    # 1. Collect all unique dataset names across all models to build standard columns
    all_datasets = set()
    model_data = {} # {model_name: {global: {}, datasets: {name: metrics}}}

    for m_dir in model_dirs:
        model_name = os.path.basename(m_dir) if m_dir != results_root else "Current"
        manifests = glob.glob(os.path.join(m_dir, "*.json"))
        
        if not manifests:
            continue

        dataset_map = {}
        metrics_sum = defaultdict(float)
        num_datasets = 0

        for manifest in manifests:
            # Clean dataset name: remove .json and path
            ds_name = os.path.basename(manifest).replace(".json", "")
            # Remove redundant suffixes if present (optional)
            ds_name = ds_name.replace("_output", "").replace("_result", "")
            
            all_datasets.add(ds_name)
            
            mets, detailed_items = calculate_metrics(manifest)
            dataset_map[ds_name] = mets
            
            if save_to_db:
                # Log to DB
                try:
                    exp_id = log_experiment(
                        model_name=model_name,
                        dataset_name=ds_name,
                        global_wer=mets["wer"],
                        global_cer=mets["cer"],
                        avg_latency=mets["avg_latency"]
                    )
                    log_inferences(exp_id, detailed_items)
                except Exception as e:
                    print(f"Failed to log to DB for {model_name}/{ds_name}: {e}")
            
            metrics_sum["wer"] += mets["wer"]
            metrics_sum["cer"] += mets["cer"]
            metrics_sum["avg_latency"] += mets["avg_latency"]
            metrics_sum["rtf"] += mets["rtf"]
            num_datasets += 1
        
        if num_datasets > 0:
            global_mets = {
                "wer": metrics_sum["wer"] / num_datasets,
                "cer": metrics_sum["cer"] / num_datasets,
                "avg_latency": metrics_sum["avg_latency"] / num_datasets,
                "rtf": metrics_sum["rtf"] / num_datasets
            }
            model_data[model_name] = {
                "global": global_mets,
                "datasets": dataset_map
            }

    # 2. Print Table
    sorted_datasets = sorted(list(all_datasets))
    
    # Headers
    headers = ["Model", "Avg Latency", "Avg WER", "Avg CER"]
    for ds in sorted_datasets:
        headers.extend([f"{ds} Lat", f"{ds} WER", f"{ds} CER"])
    
    # Format string construction (dynamic based on headers)
    # Using tabulate would be nicer but standard padding works too
    header_str = f"{headers[0]:<25} | " + " | ".join([f"{h:<12}" for h in headers[1:]])
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    for model_name, data in model_data.items():
        row = [f"{model_name:<25}"]
        
        # Globals
        g = data["global"]
        row.append(f"{g['avg_latency']:<12.4f}")
        row.append(f"{g['wer']:<12.4f}")
        row.append(f"{g['cer']:<12.4f}")
        
        # Per Dataset
        for ds in sorted_datasets:
            if ds in data["datasets"]:
                d = data["datasets"][ds]
                row.append(f"{d['avg_latency']:<12.4f}")
                row.append(f"{d['wer']:<12.4f}")
                row.append(f"{d['cer']:<12.4f}")
            else:
                row.append(f"{'N/A':<12}")
                row.append(f"{'N/A':<12}")
                row.append(f"{'N/A':<12}")
        
        print(" | ".join(row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Global Leaderboard Scores")
    parser.add_argument("--results_dir", required=True, help="Root directory containing model subfolders (e.g. results/speechbrain/d1.json)")
    parser.add_argument("--save_to_db", action="store_true", help="Save metrics to SQLite database")
    parser.add_argument("--clear_db", action="store_true", help="Clear the database before saving")
    args = parser.parse_args()
    main(args.results_dir, save_to_db=args.save_to_db, clear_db_flag=args.clear_db)
