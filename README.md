
# Universal Arabic ASR Leaderboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A comprehensive benchmark and leaderboard for Arabic Automatic Speech Recognition (ASR) models, focusing on diverse dialects and conditions.

## 🏆 Current Leaderboard

| Model | Avg Latency | Avg WER | Avg CER | Sada WER | Casablanca WER | Arabic Diacritized WER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-ASR (1.7B)** | **1.32s** | **48.8%** | **20.4%** | 50.1% | 69.8% | **26.6%** |
| **Omnilingual (1B)** | 17.8s | 41.3% | 20.4% | **44.2%** | **64.9%** | 14.8% |
| **Whisper (Large-v3)** | 117s | 49.5% | 27.1% | 57.1% | 73.4% | 17.9% |

## 🚀 Key Features

- **Multi-Model Support**: Evaluates Whisper, Qwen2.5, and Omnilingual models.
- **Diverse Datasets**:
    - **Sada**: Large-scale Saudi dialect.
    - **Casablanca**: Challenging Moroccan dialect.
    - **Arabic Diacritized**: Standard Arabic with diacritics.
- **Interactive Dashboard**: Streamlit-based dashboard to visualize results and deep-dive into specific samples.
- **Metrics**: Standard WER/CER plus Real-Time Factor (RTF) and Latency.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/open_universal_arabic_asr_leaderboard.git
    cd open_universal_arabic_asr_leaderboard
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

### Run the Dashboard
The dashboard allows you to explore the pre-computed results stored in `metrics.db`.

```bash
streamlit run dashboard.py
```

### Run Inference
To evaluate a model (e.g., Qwen) on a dataset:

```bash
python run_qwen_asr.py \
  --model_id "Qwen/Qwen2.5-ASR-1.7B" \
  --data_manifest datasets/sada_test.json \
  --output_manifest results/qwen/sada_output.json \
  --language Arabic
```

## 📂 Project Structure

- `dashboard.py`: Streamlit application code.
- `models/`: Inference scripts for different architectures.
- `datasets/`: Manifest files (JSON) for test sets.
- `results/`: Raw JSON output from inferences.
- `metrics.db`: SQLite database containing all evaluation results.
- `scripts/`: various utility scripts for scoring and database management.
