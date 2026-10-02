# 🧠 Laya CSV Evaluator

A Streamlit app to evaluate the [Laya](https://github.com/NandhaKishorM/laya) decision engine on your own CSV data — no code required.

---

## What is Laya?

[Laya](https://github.com/NandhaKishorM/laya) is a lightweight, **non-autoregressive** decision engine built on top of encoder-only transformers (ModernBERT / mmBERT). Unlike LLMs, it does not generate text — it answers **typed questions** about a piece of text in a single forward pass, making it very fast and efficient.

It supports three question types:

- **`choice`** — classifies text into one of your defined labels
- **`score`** — assigns an ordinal float value (e.g. urgency level)
- **`noul`** — returns a probability (0–1) for a yes/no signal

Laya runs **fully locally**, requires no API key, and works on CPU or GPU.

---

## Requirements

- Python **3.10 or newer**
- An NVIDIA GPU is optional but recommended (CUDA 12.x). CPU works too.

---

## Quick Start

Open a terminal inside the `Laya App/` folder, then run:

```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements_app.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## GPU acceleration (optional)

If you have an NVIDIA GPU, install PyTorch with CUDA support **before** step 2 above:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then run `pip install -r requirements_app.txt` and `streamlit run app.py` as usual.

---

## How to use

The app has a **4-step sidebar wizard**:

| Step | What to do |
|------|------------|
| **1 — Upload & Columns** | Upload your `.csv` file and select which text columns to use as input |
| **2 — Sample & Model** | Set how many rows to evaluate, batch size, random seed, and Laya checkpoint |
| **3 — Questions** | Define your typed questions (`choice`, `score`, `noul`) with criteria |
| **4 — Ground Truth** | Optionally map each question to a CSV column for accuracy metrics |

Then click **▶ Run Evaluation** to start inference.


---

## Question types

| Type | Output | Use for |
|------|--------|---------|
| `choice` | Predicted label string | Classification (category, department…) |
| `score` | Float (ordinal level) | Urgency, severity, rating |
| `noul` | Float 0–1 (P(true)) | Yes/No signals (blocking, churn risk…) |

---

## Results

After running, the main area shows:

- ⏱ Inference time and throughput
- 📊 Accuracy, Macro F1, Weighted F1 (when ground truth is set)
- 🟦 Confusion matrix heatmap
- 📈 Confidence histogram (correct vs incorrect)
- 📉 Calibration curve
- 🔢 Per-class accuracy bar chart
- 📥 Download results as CSV

<img src="image/Kaggle%20Ticket%20Open%20Source%20Data%20Result.png" width="1000"/>

<img src="image/Kaggle%20Ticket%20Open%20Source%20Data%20Resul%202%20t.png" width="1000"/>

Image: Results from kaggle open source [ticket](https://www.kaggle.com/datasets/avii3301/synthetic-itsm-ticket-dataset)

---

## Checkpoints

| Name | Params | Context | Best for |
|------|--------|---------|----------|
| `auto` | — | — | Mixed / unknown language (Router picks automatically) |
| `english` | 421M | 512 tokens | English-only text |
| `multilingual` | 322M | up to 8 192 tokens | 100+ languages |
| `typed-decisions` | 421M | 1 024 tokens | Fine-tuned typed decision workflows |
