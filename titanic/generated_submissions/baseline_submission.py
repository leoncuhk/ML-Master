
import os
import pandas as pd
import polars as pl
import sys

# Add the competition API to the path
sys.path.append('./kaggle_evaluation')
import mitsui_inference_server

NUM_TARGET_COLUMNS = 424

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """A baseline prediction function that returns a simple average."""
    predictions = pl.DataFrame({f'target_{i}': i / 1000 for i in range(NUM_TARGET_COLUMNS)})
    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == 1
    return predictions

# --- Server Execution ---
# In a real Kaggle environment, KAGGLE_IS_COMPETITION_RERUN would be set.
# For local testing, we use run_local_gateway.
inference_server = mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # The path to the competition data directory
    data_dir = '/Users/leon/Documents/Git/ml-master/mitsui-commodity-prediction-challenge/'
    inference_server.run_local_gateway((data_dir,))
