import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GROUND_TRUTH_FILE = "ground_truth.json"
MODEL_1_FILE = "model_1_output.json"
MODEL_2_FILE = "model_2_output.json"


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Similarity score required to consider a prediction a match
SIMILARITY_THRESHOLD = 0.53


# ---------------------------------------------------------
# Load JSON files
# ---------------------------------------------------------

def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


ground_truth = load_json(GROUND_TRUTH_FILE)
model_1 = load_json(MODEL_1_FILE)
model_2 = load_json(MODEL_2_FILE)


print("Loaded datasets:")
print("Ground Truth:", len(ground_truth["contradictions"]))
print("Model 1:", len(model_1["contradictions"]))
print("Model 2:", len(model_2["contradictions"]))


# ---------------------------------------------------------
# Prepare contradiction text
# ---------------------------------------------------------

def combine_text(item):
    """
    Combine Statement1, Statement2, and Reasoning into one text.

    Using all three fields gives the embedding model more context
    when comparing contradictions semantically.
    """
    return (
        f"{item['Statement1']} "
        f"{item['Statement2']} "
        f"{item['Reasoning']}"
    )


def prepare_texts(dataset):
    """
    Convert all contradiction items in a dataset into combined text.
    """
    return [
        combine_text(item)
        for item in dataset["contradictions"]
    ]


ground_truth_texts = prepare_texts(ground_truth)
model_1_texts = prepare_texts(model_1)
model_2_texts = prepare_texts(model_2)


# ---------------------------------------------------------
# Create embeddings
# ---------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

ground_truth_embeddings = embedding_model.encode(ground_truth_texts)
model_1_embeddings = embedding_model.encode(model_1_texts)
model_2_embeddings = embedding_model.encode(model_2_texts)

print("Embedding shapes:")
print("Ground Truth:", ground_truth_embeddings.shape)
print("Model 1:", model_1_embeddings.shape)
print("Model 2:", model_2_embeddings.shape)


# ---------------------------------------------------------
# Calculate cosine similarity
# ---------------------------------------------------------

similarity_matrix_model_1 = cos_sim(
    model_1_embeddings,
    ground_truth_embeddings
)

similarity_matrix_model_2 = cos_sim(
    model_2_embeddings,
    ground_truth_embeddings
)


# ---------------------------------------------------------
# Display best semantic match for each prediction
# ---------------------------------------------------------

def print_best_matches(model_name, similarity_matrix):
    """
    Print the best ground-truth match and cosine similarity score
    for each model prediction.
    """
    print(f"\nBest matches for {model_name}:")

    for prediction_index, row in enumerate(similarity_matrix):
        best_score = row.max().item()
        best_gt_index = row.argmax().item()

        print(
            f"Prediction {prediction_index + 1} -> "
            f"Ground Truth {best_gt_index + 1}, "
            f"score = {best_score:.4f}"
        )


print_best_matches("Model 1", similarity_matrix_model_1)
print_best_matches("Model 2", similarity_matrix_model_2)


# ---------------------------------------------------------
# Evaluate TP, FP, and FN
# ---------------------------------------------------------

def evaluate_model(similarity_matrix, threshold, total_ground_truth):
    """
    Evaluate model predictions against the ground truth.

    TP:
        Prediction matches a previously unmatched ground-truth item
        above the similarity threshold.

    FP:
        Prediction does not reach the threshold, or attempts to match
        a ground-truth contradiction that has already been matched.

    FN:
        Ground-truth contradictions that were never matched.
    """
    matched_ground_truth = set()

    tp = 0
    fp = 0

    for row in similarity_matrix:
        best_score = row.max().item()
        best_gt_index = row.argmax().item()

        is_above_threshold = best_score >= threshold
        is_new_match = best_gt_index not in matched_ground_truth

        if is_above_threshold and is_new_match:
            tp += 1
            matched_ground_truth.add(best_gt_index)
        else:
            fp += 1

    fn = total_ground_truth - len(matched_ground_truth)

    return tp, fp, fn


total_ground_truth = len(ground_truth["contradictions"])

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


# ---------------------------------------------------------
# Calculate precision, recall, and F1
# ---------------------------------------------------------

def calculate_metrics(tp, fp, fn):
    """
    Calculate precision, recall, and F1 score.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


m1_tp, m1_fp, m1_fn = model_1_results
m2_tp, m2_fp, m2_fn = model_2_results

m1_precision, m1_recall, m1_f1 = calculate_metrics(
    m1_tp,
    m1_fp,
    m1_fn
)

m2_precision, m2_recall, m2_f1 = calculate_metrics(
    m2_tp,
    m2_fp,
    m2_fn
)


# ---------------------------------------------------------
# Print final evaluation summary
# ---------------------------------------------------------

print("\nEvaluation Summary")
print("-" * 70)

print(
    f"{'Model':<10}"
    f"{'TP':<6}"
    f"{'FP':<6}"
    f"{'FN':<6}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<10}"
)

print("-" * 70)

print(
    f"{'Model 1':<10}"
    f"{m1_tp:<6}"
    f"{m1_fp:<6}"
    f"{m1_fn:<6}"
    f"{m1_precision:<12.2f}"
    f"{m1_recall:<12.2f}"
    f"{m1_f1:<10.2f}"
)

print(
    f"{'Model 2':<10}"
    f"{m2_tp:<6}"
    f"{m2_fp:<6}"
    f"{m2_fn:<6}"
    f"{m2_precision:<12.2f}"
    f"{m2_recall:<12.2f}"
    f"{m2_f1:<10.2f}"
)


def get_match_details(similarity_matrix, threshold):
    """
    Return detailed matching information for every prediction.
    """
    matches = []
    matched_ground_truth = set()

    for prediction_index, row in enumerate(similarity_matrix):
        best_score = row.max().item()
        best_gt_index = row.argmax().item()

        is_match = (
            best_score >= threshold
            and best_gt_index not in matched_ground_truth
        )

        if is_match:
            status = "TP"
            matched_ground_truth.add(best_gt_index)
        else:
            status = "FP"

        matches.append({
            "Prediction": prediction_index + 1,
            "Best Ground Truth": best_gt_index + 1,
            "Cosine Score": best_score,
            "Status": status
        })

    return matches