# Multilingual-Toxic-Comment-NLP

## Overview
This project fine-tunes a multilingual transformer model to classify toxic comments across multiple languages. We use a sampled subset of the Jigsaw Multilingual Toxic Comment Classification dataset to build a lightweight, memory-efficient prototype.

## Dataset
- The dataset is from the [Jigsaw Multilingual Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification).
- 1000 random samples were used to fit within hardware (Mac MPS) memory limits.
  
## Model
- Pretrained Model: `xlm-roberta-base`
- Tokenizer: Max length = 128
- Fine-tuned for 1 epoch
- Batch size: 2
- Final training loss: ~0.1785

## Environment
Libraries used:
- PyTorch
- HuggingFace Transformers
- Datasets
- Pandas, Matplotlib

## Results
The model was successfully fine-tuned on the sampled dataset, achieving a low training loss. 

## Future Work
- Fine-tune with full dataset
- Test zero-shot multilingual performance
- Try larger models like XLM-Roberta-Large

---

👤 Author

**Satwik Gudapati**

Master's Student in Information Science

University of Arizona

LinkedIn (https://www.linkedin.com/in/satwik-gudapati/)

---

🙏 Acknowledgements

Jigsaw Multilingual Challenge (Kaggle) (https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/)

Hugging Face Transformers (https://huggingface.co/docs/transformers/en/index)

scikit-learn (https://scikit-learn.org/stable/)

Wikipedia Talk Pages (https://meta.wikimedia.org/wiki/Talk_pages_project)
