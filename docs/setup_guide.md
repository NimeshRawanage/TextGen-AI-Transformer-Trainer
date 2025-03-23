#  Setup Guide for TextGen-AI-Transformer-Trainer

This guide explains how to set up, train, and run the project both in **Google Colab** and **locally**.

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

## 🧠 Features

- Fine-tune **GPT-2** or **T5** on any `.txt`, `.csv`, or `.json` dataset.
- Save model checkpoints automatically by dataset and model type.
- Easily run **inference** or launch **interactive chatbot** mode.
- Works in **Google Colab**, local environments, or servers.

---

## ✅ Recommended: Use Google Colab

1. **Fork this repo to your GitHub account**

    Go to: [github.com/NimeshRawanage/TextGen-AI-Transformer-Trainer](https://github.com/NimeshRawanage/TextGen-AI-Transformer-Trainer)  
    Then click the **Fork** button.

2. **Open the Colab Notebook**

    Open `notebooks/TextGen_Colab.ipynb` from your forked repo:  
    https://colab.research.google.com/github/**YOUR_USERNAME**/TextGen-AI-Transformer-Trainer/blob/main/notebooks/TextGen_Colab.ipynb

3. **Follow the steps inside the notebook**

    - Mount Google Drive
    - Clone your forked repo
    - Install dependencies
    - Train model using `train.py`
    - Generate text or launch chatbot using `inference.py`

---

## 💻 Run Locally (Optional)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/TextGen-AI-Transformer-Trainer.git
cd TextGen-AI-Transformer-Trainer

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

### 3. Run training

```bash
python src/train.py --model gpt2 --dataset_path datasets/sample_data.txt --epochs 3 --batch_size 4 --output_dir models/

### 4. Run inference

```bash
python src/inference.py --model_dir models/gpt2_sample_data --prompt "Hello, how are you?" --max_length 50

### 5. Run chatbot

```bash
python src/inference.py --model_dir models/gpt2_sample_data --mode chat

---

📝 Notes
Ensure your dataset is in .txt, .csv, or .json format with plain text in the first column (for .csv or .json).
Model outputs will be saved to the models/ directory automatically.
The tokenizer will be reused from the original model (gpt2, t5-small, etc.).
🙋‍♂️ Support
If you find this project useful, feel free to give it a ⭐ and share it.

---

**Author**: Nimesh Rawanage  
AI Engineer & Backend Developer  
**GitHub**: [github.com/NimeshRawanage](https://github.com/NimeshRawanage)  
**LinkedIn**: [linkedin.com/in/nimeshrawanage](https://linkedin.com/in/nimeshrawanage)

