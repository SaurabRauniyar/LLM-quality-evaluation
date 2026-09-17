import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SIMILARITY_THRESHOLD = 0.53
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load JSON files
# ---------------------------------------------------------

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


ground_truth = load_json("ground_truth.json")
model_1 = load_json("model_1_output.json")
model_2 = load_json("model_2_output.json")
my_model = load_json("my_model_output.json")


# ---------------------------------------------------------
# Prepare text
# ---------------------------------------------------------

def combine_text(item):
    return (
        f"{item['Statement1']} "
        f"{item['Statement2']} "
        f"{item['Reasoning']}"
    )


def prepare_texts(dataset):
    return [
        combine_text(item)
        for item in dataset["contradictions"]
    ]


ground_truth_texts = prepare_texts(ground_truth)
model_1_texts = prepare_texts(model_1)
model_2_texts = prepare_texts(model_2)
my_model_texts = prepare_texts(my_model)


# ---------------------------------------------------------
# Create embeddings
# ---------------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

ground_truth_embeddings = embedding_model.encode(ground_truth_texts)
model_1_embeddings = embedding_model.encode(model_1_texts)
model_2_embeddings = embedding_model.encode(model_2_texts)
my_model_embeddings = embedding_model.encode(my_model_texts)


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

similarity_matrix_model_1 = cos_sim(
    model_1_embeddings,
    ground_truth_embeddings
)

similarity_matrix_model_2 = cos_sim(
    model_2_embeddings,
    ground_truth_embeddings
)

similarity_matrix_my_model = cos_sim(
    my_model_embeddings,
    ground_truth_embeddings
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(similarity_matrix, threshold, total_ground_truth):
    matched_ground_truth = set()

    tp = 0
    fp = 0

    for row in similarity_matrix:
        best_score = row.max().item()
        best_gt_index = row.argmax().item()

        if (
            best_score >= threshold
            and best_gt_index not in matched_ground_truth
        ):
            tp += 1
            matched_ground_truth.add(best_gt_index)
        else:
            fp += 1

    fn = total_ground_truth - len(matched_ground_truth)

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


total_ground_truth = len(ground_truth["contradictions"])


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

model_1_results = evaluate_model(
    similarity_matrix_model_1,
    SIMILARITY_THRESHOLD,
    total_ground_truth
)

model_2_results = evaluate_model(
    similarity_matrix_model_2,
    SIMILARITY_THRESHOLD,
    total_ground_truth
)

my_model_results = evaluate_model(
    similarity_matrix_my_model,
    SIMILARITY_THRESHOLD,
    total_ground_truth
)


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

m1_tp, m1_fp, m1_fn = model_1_results
m2_tp, m2_fp, m2_fn = model_2_results
my_tp, my_fp, my_fn = my_model_results

m1_precision, m1_recall, m1_f1 = calculate_metrics(
    m1_tp, m1_fp, m1_fn
)

m2_precision, m2_recall, m2_f1 = calculate_metrics(
    m2_tp, m2_fp, m2_fn
)

my_precision, my_recall, my_f1 = calculate_metrics(
    my_tp, my_fp, my_fn
)


# ---------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------

print("\nEvaluation Summary")
print("-" * 75)

print(
    f"{'Model':<15}"
    f"{'TP':<6}"
    f"{'FP':<6}"
    f"{'FN':<6}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<10}"
)

print("-" * 75)

print(
    f"{'Model 1':<15}"
    f"{m1_tp:<6}"
    f"{m1_fp:<6}"
    f"{m1_fn:<6}"
    f"{m1_precision:<12.2f}"
    f"{m1_recall:<12.2f}"
    f"{m1_f1:<10.2f}"
)

print(
    f"{'Model 2':<15}"
    f"{m2_tp:<6}"
    f"{m2_fp:<6}"
    f"{m2_fn:<6}"
    f"{m2_precision:<12.2f}"
    f"{m2_recall:<12.2f}"
    f"{m2_f1:<10.2f}"
)

print(
    f"{'My Task 1':<15}"
    f"{my_tp:<6}"
    f"{my_fp:<6}"
    f"{my_fn:<6}"
    f"{my_precision:<12.2f}"
    f"{my_recall:<12.2f}"
    f"{my_f1:<10.2f}"
)