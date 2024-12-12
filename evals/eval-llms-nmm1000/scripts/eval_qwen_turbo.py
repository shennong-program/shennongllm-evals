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
LOG_FILE_PATH = os.path.join(LOGS_DIR, "eval_qwen_turbo.log")

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


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

models = [
    "qwen-turbo-2024-09-19",
]

for model in models:
    get_llm_answers_by_nmm_question_ids_and_save_to_csv(
        nmm_question_ids=get_all_nmm_question_ids(),
        model=model,
        client=client,
        csv_file_path=os.path.join(
            RESULTS_DIR, f"nmm1000-questions-results-{model}.csv"
        ),
        overwrite=True,
    )
