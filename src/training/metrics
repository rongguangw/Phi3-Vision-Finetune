import Levenshtein
import nltk
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import evaluate

em_metric = evaluate.load("exact_match")
f1_metric = evaluate.load("f1")


def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(
        predicted_answers
    ), 'Length of ground_truth and predicted_answers must match.'

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == '':
            print('Warning: Skipped an empty prediction.')
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N

def compute_metrics(pred):
    references = pred.label_ids
    generated_texts = pred.predictions.argmax(-1)
    #return {'similarity': average_normalized_levenshtein_similarity(references, generated_texts)}

    # Calculate Exact Match (EM)
    #em = sum([1 if p == l else 0 for p, l in zip(generated_texts, references)]) / len(references)

    # Calculate F1-score
    #f1 = f1_score(references, generated_texts, average='macro')

    return {
        'exact_match': em_metric.compute(predictions=generated_texts, references=references),
        'f1': f1_metric.compute(predictions=generated_texts, references=references)
    }
