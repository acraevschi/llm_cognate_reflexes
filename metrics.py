import Levenshtein

def calculate_ned(pred_str, ref_str):
    """
    Calculates Normalized Edit Distance between two strings.
    Removes spaces to compare phoneme sequences regardless of segmentation.
    0.0 = Perfect match, 1.0 = Completely different
    """
    # Ensure inputs are strings and strip surrounding whitespace
    p = str(pred_str).strip()
    r = str(ref_str).strip()
    
    # Remove internal spaces (e.g., "t a t a" -> "tata")
    # This ensures consistency if the model varies spacing
    p = p.replace(" ", "")
    r = r.replace(" ", "")
    
    # Handle empty reference case
    if len(r) == 0:
        return 1.0 if len(p) > 0 else 0.0
        
    dist = Levenshtein.distance(p, r)
    max_len = max(len(p), len(r))
    
    return dist / max_len if max_len > 0 else 0.0

def compute_batch_ned(predictions, references):
    """
    Computes average NED for a list of prediction/reference strings.
    """
    scores = []
    
    for pred, ref in zip(predictions, references):
        scores.append(calculate_ned(pred, ref))
                
    return sum(scores) / len(scores) if scores else 0.0