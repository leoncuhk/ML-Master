
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
import numpy as np
import sys

def calculate_spearman_correlation(predictions: pl.DataFrame, ground_truth: pl.DataFrame) -> float:
    """
    Calculates the mean Spearman rank correlation between predictions and ground truth.
    """
    correlations = []
    for col in predictions.columns:
        if col.startswith('target_'):
            pred_series = predictions[col].to_pandas()
            gt_series = ground_truth[col].to_pandas()
            corr, _ = spearmanr(pred_series, gt_series)
            if not np.isnan(corr):
                correlations.append(corr)
    return np.mean(correlations)

def main():
    """
    Main function to calculate the competition metric from a submission file.
    """
    if len(sys.argv) != 2:
        print("Usage: python local_scorer.py <path_to_submission.parquet>")
        sys.exit(1)

    submission_path = sys.argv[1]
    
    # Define the path to the ground truth labels
    # This assumes the script is run from the project root
    ground_truth_path = 'mitsui-commodity-prediction-challenge/train_labels.csv'

    try:
        predictions = pl.read_parquet(submission_path)
        ground_truth = pl.read_csv(ground_truth_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Align the ground truth data with the predictions
    # The local gateway uses the last 90 days of the training set for the public test set
    date_ids_in_submission = predictions['date_id'].unique()
    ground_truth = ground_truth.filter(pl.col('date_id').is_in(date_ids_in_submission))

    # Drop date_id for correlation calculation
    predictions = predictions.drop('date_id')
    ground_truth = ground_truth.drop('date_id')

    mean_corr = calculate_spearman_correlation(predictions, ground_truth)
    std_corr = np.std(correlations) # This is a simplification, as we don't have the full set of correlations

    # The competition metric is mean/std, but for local scoring, mean correlation is a good proxy
    # A more advanced implementation would need to handle the std dev calculation properly
    # For now, we will use the mean correlation as the score.
    score = mean_corr
    if std_corr > 0:
        score = mean_corr / std_corr

    print(f"Score: {score}")

if __name__ == "__main__":
    main()

