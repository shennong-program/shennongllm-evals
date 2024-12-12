import logging
import os

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NMM1000_QUESTIONS_PATH = os.path.join(
    CURRENT_DIR, "../dataset/nmm1000/nmm1000-questions.csv"
)

RESULTS_DIR = os.path.join(CURRENT_DIR, "../results")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_nmm1000_questions(
    nmm1000_questions_path: str = NMM1000_QUESTIONS_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(nmm1000_questions_path)
    required_columns = [
        "nmm_question_id",
        "nmm_question_pair_id",
        "wbk",
        "type",
        "lang",
        "nmm_question",
        "background_knowledge",
        "expected_answer",
    ]

    assert all(col in df.columns for col in required_columns)

    return df


NMM1000_QUESTIONS_DF = load_nmm1000_questions()


def expand_nmm1000_questions_df(df: pd.DataFrame) -> pd.DataFrame:
    # create 2 new columns for storing the llm answers
    new_columns = {
        # column name: default value
        "llm_answer": "",
        "correct": False,
    }
    df = df.assign(**new_columns)
    return df


def check_llm_answer_correctness(
    llm_answer: str,
    expected_answer: str,
) -> bool:
    # if the expected answer is in the llm answers, then the answer is correct. Ignore case
    return expected_answer.lower() in llm_answer.lower()


def check_llm_answers_correctness(
    nmm1000_df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = ["expected_answer", "llm_answer", "correct"]
    assert all(col in nmm1000_df.columns for col in required_columns)
    # if the expected answer is in the llm answers, then the answer is correct. Ignore case
    nmm1000_df["correct"] = nmm1000_df.apply(
        lambda row: check_llm_answer_correctness(
            row["llm_answer"], row["expected_answer"]
        ),
        axis=1,
    )
    return nmm1000_df


def save_nmm1000_llm_results(
    llm_name: str,
    nmm1000_df: pd.DataFrame,
) -> None:
    results_path = os.path.join(RESULTS_DIR, f"nmm1000-{llm_name}-results.csv")
    columns_to_save = [
        "nmm_question_id",
        "wbk",
        "expected_answer",
        "llm_answer",
        "correct",
    ]
    nmm1000_df[columns_to_save].to_csv(results_path, index=False)


def evaluate_nmm1000_llm_results(
    results: pd.DataFrame,
    txt_file_path: str | None = None,
) -> None:
    # Check if the results df has the correct columns
    required_columns = ["nmm_question_id", "wbk", "correct"]
    assert all(col in results.columns for col in required_columns)

    # Check the data types of `wbk` and `correct` columns are boolean
    assert results["wbk"].dtype == bool
    assert results["correct"].dtype == bool

    # Report the accuracy of the LLM for both `wobk` and `wbk` questions
    # Log number of both `wobk` and `wbk` questions
    wobk_results = results[~results["wbk"]]  # ~ is the negation operator
    wbk_results = results[results["wbk"]]
    num_wobk = wobk_results.shape[0]
    num_wbk = wbk_results.shape[0]
    logging.info("Number of wobk questions: %d", num_wobk)
    logging.info("Number of wbk questions: %d", num_wbk)

    # Log accuracy of the LLM for both `wobk` and `wbk` questions
    wobk_accuracy = wobk_results["correct"].mean()
    wbk_accuracy = wbk_results["correct"].mean()
    logging.info("Accuracy of wobk questions: %f", wobk_accuracy)
    logging.info("Accuracy of wbk questions: %f", wbk_accuracy)

    if txt_file_path:
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(f"Number of wobk questions: {num_wobk}\n")
            f.write(f"Number of wbk questions: {num_wbk}\n")
            f.write(f"Accuracy of wobk questions: {wobk_accuracy}\n")
            f.write(f"Accuracy of wbk questions: {wbk_accuracy}\n")


def get_all_nmm_question_ids() -> list[str]:
    return NMM1000_QUESTIONS_DF["nmm_question_id"].tolist()


def get_openai_completion(question: str, model: str, client: OpenAI) -> str:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        return str(completion.choices[0].message.content)
    except Exception as e:
        logging.error("Error while getting completion: %s", e)
        return ""


def get_nmm_question_id_llm_answer(
    nmm_question_id: str,
    model: str,
    client: OpenAI,
) -> str:
    nmm_question = NMM1000_QUESTIONS_DF.loc[
        NMM1000_QUESTIONS_DF["nmm_question_id"] == nmm_question_id, "nmm_question"
    ].values[0]
    nmm_question = str(nmm_question)
    return get_openai_completion(nmm_question, model, client)


class LlmResultRow(BaseModel):
    nmm_question_id: str
    nmm_question_pair_id: str
    wbk: bool
    expected_answer: str
    llm_answer: str
    correct: bool


def get_llm_answers_by_nmm_question_ids_and_save_to_csv(
    nmm_question_ids: list[str],
    model: str,
    client: OpenAI,
    csv_file_path: str,
    overwrite: bool = False,
) -> None:
    """
    Dynamically save the LLM answer when every question in `nmm_question_ids` is asked to the LLM. Append the last answer to the final line of the CSV file.
    The CSV should use the keys of the `LlmResultRow` class as the header.
    """
    if os.path.exists(csv_file_path) and not overwrite:
        raise FileExistsError(
            f"The file {csv_file_path} already exists. Set `overwrite=True` to overwrite the file."
        )

    with open(csv_file_path, "w", encoding="utf-8"):
        # Write header first
        header = list(LlmResultRow.model_fields.keys())
        pd.DataFrame(columns=header).to_csv(csv_file_path, index=False)

        for nmm_question_id in nmm_question_ids:
            logging.info("Processing question: %s", nmm_question_id)
            # Get required data
            question_data = NMM1000_QUESTIONS_DF[
                NMM1000_QUESTIONS_DF["nmm_question_id"] == nmm_question_id
            ].iloc[0]

            llm_answer = get_nmm_question_id_llm_answer(nmm_question_id, model, client)
            correct = check_llm_answer_correctness(
                llm_answer, question_data["expected_answer"]
            )

            # Create row data
            row_data = LlmResultRow(
                nmm_question_id=nmm_question_id,
                nmm_question_pair_id=question_data["nmm_question_pair_id"],
                wbk=question_data["wbk"],
                expected_answer=question_data["expected_answer"],
                llm_answer=llm_answer,
                correct=correct,
            )

            # Append row to CSV
            pd.DataFrame([row_data.model_dump()]).to_csv(
                csv_file_path, mode="a", header=False, index=False
            )
