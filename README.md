# LLM Quality Evaluation

This project evaluates two LLM outputs against a ground-truth set of 10 planted contradictions from a technical specification.

## Files

- `ground_truth.json` — reference contradictions
- `model_1_output.json` — Model 1 predictions
- `model_2_output.json` — Model 2 predictions
- `evaluate.py` — evaluation code
- `report.txt` — evaluation results and interpretation
- `requirements.txt` — Python dependencies

## Method

The evaluation uses semantic similarity to match model predictions with ground-truth contradictions.

Each contradiction combines:
- Statement1
- Statement2
- Reasoning

The combined text is converted into embeddings using:

`all-MiniLM-L6-v2`

Cosine similarity is then used to compare predictions with ground-truth items.

A similarity threshold of `0.53` is used to decide whether a prediction is considered a match.

## Metrics

The script calculates:

- True Positives
- False Positives
- False Negatives
- Precision
- Recall
- F1 Score

## Setup

Create and activate a virtual environment.

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate


