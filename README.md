# Multilingual-Toxic-Coment-NLP
A multilingual NLP project for detecting toxic user comments across languages using transformer models like XLM-RoBERTa.

## 📌 Project Overview

This project tackles the task of **toxic comment classification** across multiple languages using transformer-based models. The goal is to build and evaluate models that can accurately detect toxic content in user comments in languages such as English, Spanish, Italian, Russian, Turkish, and more.

The dataset comes from the **[Jigsaw Multilingual Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification)**, hosted on Kaggle. This work aims to demonstrate how multilingual transformer models can generalize well across languages and how transfer learning can aid in zero-shot performance.

---

## 🎯 Objectives

- Build a robust multilingual classifier to detect toxic comments.
- Fine-tune transformer models like `xlm-roberta-base` and `bert-base-multilingual-cased`.
- Compare zero-shot performance and language-specific fine-tuning.
- Evaluate model generalization across languages and label distributions.

---

## 📚 Dataset

The dataset is from the [Jigsaw Multilingual Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification).

- `jigsaw-toxic-comment-train.csv`: English comments from Wikipedia (labeled).
- `validation.csv`: Multilingual validation set (labeled).
- `test.csv`: Multilingual test set (unlabeled).
- `test_labels.csv`: Ground-truth for test (released post-competition).
- Pre-tokenized versions (`*-seqlen128.csv`) are also available but this project handles raw text tokenization for flexibility.

You must accept the competition rules on Kaggle to access the data. Use the Kaggle CLI or manual download.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Clean text: remove HTML tags, special characters, and normalize Unicode
- Tokenize using the appropriate tokenizer (e.g., XLM-R tokenizer)
- Balance classes for training

### 2. Modeling
- Fine-tune `xlm-roberta-base` using Hugging Face Transformers
- Baseline comparison with `bert-base-multilingual-cased`
- Experiments:
  - Language-specific training
  - Multilingual training
  - Zero-shot evaluation (e.g., train on English, test on other languages)

### 3. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Per-language breakdown
- Confusion matrix

---

## 🧪 Dependencies

Use the provided `requirements.txt` file.

Main packages:
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `torch` or `tensorflow`

---

👤 Author

Satwik Gudapati

Master's Student in Information Science

University of Arizona

LinkedIn (https://www.linkedin.com/in/satwik-gudapati/)

---

🙏 Acknowledgements

Jigsaw Multilingual Challenge (Kaggle) (https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/)

Hugging Face Transformers (https://huggingface.co/docs/transformers/en/index)

scikit-learn (https://scikit-learn.org/stable/)

Wikipedia Talk Pages (https://meta.wikimedia.org/wiki/Talk_pages_project)

