def calculate_improved_precision(precision, consecutive_count):
    fpr = 1 - precision
    improved_fpr = fpr ** consecutive_count
    improved_precision = 1 - improved_fpr
    return improved_precision

# Example usage
current_precision = 0.13  # Current precision of the 'aggression' class
consecutive_predictions_needed = 17
new_precision = calculate_improved_precision(current_precision, consecutive_predictions_needed)
print(f"Estimated new precision with {consecutive_predictions_needed} consecutive predictions: {new_precision:.2f}")
