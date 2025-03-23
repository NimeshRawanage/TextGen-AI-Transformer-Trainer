# TextGen-AI-Transformer-Trainer

A lightweight Python framework for fine-tuning transformer models (GPT-2 / T5) on your own custom text datasets.  
Supports both **text generation** and **interactive chatbot** mode using Hugging Face Transformers.  
Works seamlessly on **Google Colab** and in **local environments**.

---

## Features

- ✅ Fine-tune **GPT-2** or **T5** on `.txt`, `.csv`, or `.json` files.
- ✅ Automatically saves models by dataset + model name.
- ✅ Built-in **inference** and **chat** modes.
- ✅ Supports Hugging Face Datasets and custom files.
- ✅ Runs on **Google Colab**, local machines, or remote servers.
- ✅ Clean modular code: `preprocess.py`,`train.py`,`inference.py`.

---

## 📁 Project Structure

```
TextGen-AI-Transformer-Trainer/
├── datasets/                 # Sample & custom dataset files
├── docs/                     # Documentation (setup guide, etc.)
├── logs/                     # Training logs directory
├── models/                   # Trained model output directory
├── notebooks/                # Jupyter / Colab notebooks
│   └── TextGen_Colab.ipynb   # Colab-friendly training/inference notebook
├── src/                      # Main training and inference code
│   ├── train.py              # Training script
│   ├── inference.py          # Inference & chatbot script
│   └── preprocess.py         # Data loading and preprocessing logic
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and instructions
└── .gitignore                # Git ignore rules
```

---

## 🔧 Setup Options

### ▶️ Option 1: Google Colab (Recommended).

No setup needed — everything runs in the cloud.

1. **Fork** this repository to your own GitHub.
2. Open the Colab Notebook (notebooks/TextGen_Colab.ipynb).
```
https://colab.research.google.com/github/YOUR_USERNAME/TextGen-AI-Transformer-Trainer/blob/main/notebooks/TextGen_Colab.ipynb
```
3. Follow the instructions to:
   - Mount Google Drive
   - Clone your fork
   - Train a model using your dataset
   - Run inference or launch chat mode

---

### 💻 Option 2: Run Locally

#### Step 1: Clone the repository
```
git clone https://github.com/NimeshRawanage/TextGen-AI-Transformer-Trainer.git
cd TextGen-AI-Transformer-Trainer
```
Step 2: Install dependencies
```
pip install -r requirements.txt
```
Step 3: Train the model
```
python src/train.py \
  --model gpt2 \
  --dataset_path datasets/sample_data.txt \
  --epochs 3 \
  --batch_size 4 \
  --output_dir models/
```
Step 4: Run inference

```
python src/inference.py \
  --model_dir models/gpt2_sample_data \
  --prompt "Once upon a time" \
  --max_length 50
```

Step 5: Launch chatbot mode

```
python src/inference.py \
  --model_dir models/gpt2_sample_data \
  --mode chat
```

## 📄 Supported Dataset File Formats

.txt → Each line as a training sample

.csv or .json → First column as text input

---

## 📚 Documentation

A full setup guide is available in docs/setup_guide.md

---

## 📌 License

This project is licensed under the MIT License.

---
⭐ If you find this helpful, please star the repository and share it with others!

---

**Author**: Nimesh Rawanage  
AI Engineer & Backend Developer  
**GitHub**: [github.com/NimeshRawanage](https://github.com/NimeshRawanage)  
**LinkedIn**: [linkedin.com/in/nimeshrawanage](https://linkedin.com/in/nimeshrawanage)

