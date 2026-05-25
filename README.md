# PRODIGY_GA_01: GPT-2 Recipe Generator

A concise implementation of **Supervised Fine-Tuning (SFT)** on a pre-trained **distilgpt2** model, optimized for style-specific text generation.

## 📌 Project Overview
This project demonstrates how to adapt a large-scale language model for specific domains using transfer learning. The codebase includes end-to-end documentation explaining the deep learning mechanics of Transformers and the fine-tuning process.

## 🚀 Key Features
* **Fine-Tuning Engine:** Uses Hugging Face `Trainer` API to adapt GPT-2 to custom recipe structures.
* **Architecture:** Leverages the Transformer mechanism and self-attention to maintain logical output.
* **GPU-Optimized:** Configured for T4 GPU execution with FP16 precision.
* **Constraint Logic:** Implements `temperature`, `top_k`, and `top_p` sampling to ensure domain-specific accuracy.

## 🧠 Concepts Included
* **Self-Attention:** Understanding how the model captures context across tokens.
* **Transfer Learning:** Applying general-purpose weights to a specialized dataset.
* **Loss Monitoring:** Using epoch-based training to measure and minimize model error.

---
*Internship project for the Prodigy InfoTech Generative AI Track.*
