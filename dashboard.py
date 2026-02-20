
import streamlit as st
import sqlite3
import pandas as pd
import os
import json

DB_NAME = "metrics.db"

# Page Config
st.set_page_config(page_title="ASR Leaderboard Dashboard", layout="wide", page_icon="📊")

# Title and Style
st.title("📊 Universal Arabic ASR Leaderboard")
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #31333F;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame(), pd.DataFrame()
    
    conn = sqlite3.connect(DB_NAME)
    
    # Experiments
    df_exp = pd.read_sql_query("SELECT * FROM experiments", conn)
    
    # Inferences (High level fetch, optimization: maybe don't load ALL rows at once if huge?)
    # For now, let's load specific deep dive data on demand to avoid memory issues.
    # We just return exp data here.
    conn.close()
    return df_exp

@st.cache_data(ttl=60)
def load_inferences(model_name, dataset_name):
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT 
            i.audio_filepath, 
            i.ground_truth, 
            i.prediction, 
            i.wer, 
            i.cer, 
            i.inference_time, 
            i.rtf
        FROM inferences i
        JOIN experiments e ON i.experiment_id = e.id
        WHERE e.model_name = ? AND e.dataset_name = ?
        ORDER BY i.wer DESC
    """
    df = pd.read_sql_query(query, conn, params=(model_name, dataset_name))
    conn.close()
    return df

df_exp = load_data()

if df_exp.empty:
    st.warning(f"No database found at {DB_NAME}. Please run scoring script first.")
    st.stop()

# -- Sidebar --
st.sidebar.header("Navigation")
view_mode = st.sidebar.radio("View Mode", ["🏆 Leaderboard Overview", "🔍 Deep Dive Analysis"])

# -- Main Content --

if view_mode == "🏆 Leaderboard Overview":
    st.header("Leaderboard Summary")
    
    # Metrics Summary
    col1, col2, col3 = st.columns(3)
    best_wer = df_exp['global_wer'].min()
    best_model = df_exp.loc[df_exp['global_wer'].idxmin()]['model_name']
    
    with col1:
        st.metric("Total Experiments", len(df_exp))
    with col2:
        st.metric("Best WER", f"{best_wer:.4f}", help=f"Achieved by {best_model}")
    with col3:
        st.metric("Avg Latency", f"{df_exp['avg_latency'].mean():.4f}s")
        
    st.subheader("Global Results Table")
    # Display table
    st.dataframe(
        df_exp[['model_name', 'dataset_name', 'global_wer', 'global_cer', 'avg_latency', 'timestamp']]
        .sort_values(by='global_wer'),
        use_container_width=True,
        column_config={
            "global_wer": st.column_config.NumberColumn("WER", format="%.4f"),
            "global_cer": st.column_config.NumberColumn("CER", format="%.4f"),
            "avg_latency": st.column_config.NumberColumn("Latency (s)", format="%.4f"),
            "timestamp": st.column_config.DatetimeColumn("Run Date", format="D MMM YYYY, h:mm a"),
        }
    )

elif view_mode == "🔍 Deep Dive Analysis":
    st.header("Deep Dive Analysis")
    
    # Dropdowns for selection
    models = df_exp['model_name'].unique().tolist()
    selected_model = st.sidebar.selectbox("Select Model", models)
    
    datasets = df_exp[df_exp['model_name'] == selected_model]['dataset_name'].unique().tolist()
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)
    
    if selected_model and selected_dataset:
        st.markdown(f"### Results for **{selected_model}** on **{selected_dataset}**")
        
        # Load specific results
        df_inf = load_inferences(selected_model, selected_dataset)
        
        if not df_inf.empty:
            # Metrics for this run (Calculated Globally)
            # Helper for Levenshtein (Simple implementation to avoid heavy dependency if not present)
            # Or better, use basic accumulation if we trust individual counts? 
            # We don't have individual counts in DB, only rates. 
            # So we MUST calculate from strings.
            
            import numpy as np

            def calculate_edit_distance(ref, hyp):
                # Simple Levenshtein for WER (words) and CER (chars)
                # Optimized version or use library if available?
                # For dashboard responsiveness, pure python might be slow for 6k items if we do it here.
                # BUT, we only do it on load.
                
                # To be fast, let's try to use jiwer/editdistance if installed?
                # User has 'jiwer' in requirements.txt? checking...
                # 'jiwer' is in the environment (part of 'speechbrain' usually).
                # Let's try importing jiwer.
                try:
                    import jiwer
                    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)
                except ImportError:
                    # Fallback to crude approximation or 0
                    return 0.0, 0.0

            # Actually, to get valid Global WER, we need:
            # Sum(Errors) / Sum(Reference Length)
            # jiwer.compute_measures(truth, hypothesis) gives this.
            
            import jiwer
            
            # Prepare lists
            refs = df_inf['ground_truth'].tolist()
            preds = df_inf['prediction'].tolist()
            
            # Normalize for fair comparison (dashboard shouldn't re-normalize if DB is normalized, 
            # but let's assume DB strings are what we want to Score).
            # The DB strings are already normalized in the script before saving.
            
            # Calculate Global Metrics
            try:
                wer_measures = jiwer.compute_measures(refs, preds)
                global_wer = wer_measures['wer']
                
                # For CER, jiwer doesn't do CER natively in compute_measures well without transform?
                # actually it does usually. 
                # Let's use jiwer.cer(refs, preds) but that returns mean or global?
                # jiwer.cer is usually Global.
                global_cer = jiwer.cer(refs, preds)
                
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
                global_wer = 0.0
                global_cer = 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Samples", len(df_inf))
            c2.metric("Global WER", f"{global_wer:.4f}", help="Total Errors / Total Words")
            c3.metric("Global CER", f"{global_cer:.4f}", help="Total Char Errors / Total Chars")
            c4.metric("Avg RTF", f"{df_inf['rtf'].mean():.4f}")
            
            st.divider()
            
            # Search/Filter
            search_query = st.text_input("Detailed Search (Filename, Reference, or Prediction)", "")
            
            if search_query:
                mask = (
                    df_inf['audio_filepath'].astype(str).str.contains(search_query, case=False) |
                    df_inf['ground_truth'].astype(str).str.contains(search_query, case=False) |
                    df_inf['prediction'].astype(str).str.contains(search_query, case=False)
                )
                df_display = df_inf[mask]
            else:
                df_display = df_inf
            
            # Show Table
            st.dataframe(
                df_display, 
                use_container_width=True,
                column_config={
                     "wer": st.column_config.NumberColumn("WER", format="%.2f"),
                     "cer": st.column_config.NumberColumn("CER", format="%.2f"),
                }
            )
            
            # Audio Playback Feature
            st.write("### 🎧 Audio Playback")
            selected_row_idx = st.number_input("Enter Row Number to Inspect/Play", min_value=0, max_value=len(df_display)-1, value=0, step=1)
            
            if 0 <= selected_row_idx < len(df_display):
                row = df_display.iloc[selected_row_idx]
                st.write(f"**File:** `{row['audio_filepath']}`")
                
                # Verify path existence
                # The paths in DB are often relative or formatted like 'datasets/...'
                # We need to make sure we can find them relative to the CWD
                if os.path.exists(row['audio_filepath']):
                    st.audio(row['audio_filepath'])
                else:
                    # Stream from Hugging Face if local file isn't found (for Streamlit Cloud)
                    hf_dataset = "Nooh/arabic-asr-audio"
                    # Assume path is stored as "datasets/...". We uploaded them directly inside "datasets/..." on HF
                    audio_url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/{row['audio_filepath']}"
                    
                    try:
                        st.audio(audio_url)
                        st.caption(f"Streaming from cloud: `{audio_url}`")
                    except Exception as e:
                        st.error(f"Failed to load audio from cloud. Ensure the file exists in the Hugging Face dataset: `{row['audio_filepath']}`")
                    
                st.markdown("**Reference:**")
                st.info(row['ground_truth'])
                st.markdown("**Prediction:**")
                if row['wer'] > 0:
                    st.error(row['prediction']) # Highlight error
                else:
                    st.success(row['prediction']) # Green if perfect match
            
        else:
            st.info("No inference details found for this run.")
