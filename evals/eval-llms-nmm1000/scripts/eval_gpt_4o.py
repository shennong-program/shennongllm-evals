import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .nmm1000_helper import (
    get_all_nmm_question_ids,
    get_llm_answers_by_nmm_question_ids_and_save_to_csv,
)

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "../results")

LOGS_DIR = os.path.join(CURRENT_DIR, "../logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "eval_gpt_4o.log")

# Configure logging and save logs to a file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.handlers = []  # Clear existing handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
]

for model in MODELS:
    get_llm_answers_by_nmm_question_ids_and_save_to_csv(
        nmm_question_ids=get_all_nmm_question_ids(),
        model=model,
        client=client,
        csv_file_path=os.path.join(
            RESULTS_DIR, f"nmm1000-questions-results-{model}.csv"
        ),
        overwrite=True,
    )
