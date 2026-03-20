# 🚀 Fine-Tune: LLM Supervised Fine-Tuning Pipeline

A modular and efficient pipeline for supervised fine-tuning (SFT) of Large Language Models (LLMs) using `PEFT` (LoRA), `transformers`, and `trl`. Optimized for **Apple Silicon (MPS)** and tracked with **MLflow**.

## ✨ Key Features

- **🎯 Supervised Fine-Tuning (SFT)**: Built using `trl` SFTTrainer for streamlined instruction tuning.
- **🛡️ Parameter-Efficient Fine-Tuning (PEFT)**: Full support for **LoRA** to minimize training costs and memory usage.
- **🧊 4-Bit Quantization**: Powered by `bitsandbytes` to run large models on consumer hardware.
- **📊 MLflow Tracking**: Log training progress, metrics, and models for easy experiment management.
- **🏗️ Apple Silicon Optimized**: Pre-configured to use `mps` acceleration on Mac.
- **📦 Containerized Ready**: Includes `docker-compose.yml` for isolated environments.

---

## 🏗️ Project Structure

```text
.
├── main.py               # Application entry point
├── src/
│   ├── config/           # Pydantic configuration and models
│   │   ├── models/       # Model-specific configs (LoRA, SFT)
│   │   └── settings.py   # Global app settings (MLflow, HF paths)
│   ├── data/             # Dataset loading and preprocessing
│   └── scripts/          # Core training scripts (SFT)
├── data/                 # Raw/local data storage
├── mlflow_data/          # MLflow artifacts and metadata
├── pyproject.toml        # Dependency management (uv)
└── docker-compose.yml    # Container orchestration
```

---

## 🛠️ Setup & Installation

This project uses `uv` for lightning-fast dependency management.

### 1. Clone the repository
```bash
git clone <repository-url>
cd finetune
```

### 2. Install Dependencies
Make sure you have [uv](https://github.com/astral-sh/uv) installed:
```bash
uv sync
```

### 3. Environment Configuration
The project uses `dotenv` for settings. Create a `.env` file in the **`src/`** directory:
```env
HF_DATASET_PATH=argilla/news-summary
HF_MODEL_NAME=HuggingFaceTB/SmolLM-360M-Instruct
MLFLOW_URI=http://mlflow:5000  # Use 'mlflow' for Docker, 'localhost' for local
MLFLOW_EXPERIMET=news_summary
```

---

## 🚀 Usage

### Local Development (uv)
To start the fine-tuning process directly:
```bash
uv run main.py
```

### Dockerized Environment
Run the entire stack including MLflow and the Trainer:
```bash
docker-compose up --build
```
This will start:
- **MLflow Server**: http://localhost:5000
- **SFT Trainer**: Automatically starts training when MLflow is ready.

---

---

## ⚙️ Configuration

- **LoRA Support**: Customize rank (r=16), alpha (32), and dropout (0.05) in `src/config/models/peft/lora.py`.
- **Training Hyperparameters**: Adjust batch sizes, epochs (default: 1), and learning rate (default: 2e-4) in `src/config/models/sft.py`.
- **Hardware Acceleration**: 
  - **Mac**: Optimized for `mps` with `device_map="auto"`.
  - **CPU-Fallback**: Currently configured for high-compatibility testing with `use_cpu=True` in `sft_config`. Change this in `src/config/models/sft.py` for full GPU performance.
- **Default Models**: The pipeline is pre-configured for the `HuggingFaceTB/SmolLM-360M-Instruct` model and `argilla/news-summary` dataset.

---

## 📊 Monitoring with MLflow

Track your experiments with:
```bash
# Set your MLflow tracking URI in .env (defaults to news_summary experiment)
uv run mlflow ui --backend-store-uri ./mlflow_data
```

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
