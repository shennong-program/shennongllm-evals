import os

import pandas as pd

from .nmm1000_helper import evaluate_nmm1000_llm_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "../results")

models = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "qwen-turbo-2024-09-19",
    "qwen2-7b",
    "llama-3.1-8b",
]
for model in models:
    df = pd.read_csv(
        os.path.join(RESULTS_DIR, f"nmm1000-questions-results-{model}.csv")
    )
    evaluate_nmm1000_llm_results(
        df,
        txt_file_path=os.path.join(
            RESULTS_DIR, f"nmm1000-questions-accuracy-{model}.txt"
        ),
    )
