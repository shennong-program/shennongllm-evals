import argparse
import json
import logging
import os

import pandas as pd
import vllm
from tqdm import tqdm
from transformers import AutoTokenizer

from .nmm1000_helper import (
    check_llm_answer_correctness,
    evaluate_nmm1000_llm_results,
    load_nmm1000_questions,
)


def main(args):
    # Set up logging
    os.makedirs(args.logs_dir, exist_ok=True)
    log_file_path = os.path.join(args.logs_dir, args.log_file_name)
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load input queries
    input_file_path = args.input_file_path
    df = load_nmm1000_questions(nmm1000_questions_path=input_file_path)

    dfdata = []

    for i in range(len(df)):
        dict_data = df.iloc[i].to_dict()
        dfdata.append(dict_data)

    output_file_name = args.output_file_name

    os.makedirs(args.results_dir, exist_ok=True)

    output_file_path = os.path.join(args.results_dir, output_file_name)
    print(f"output_file_path: {output_file_path}")

    # load saved samples
    dedup_set = set()
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                dedup_set.add(line_data["nmm_question_id"])
            print(f"len(dedup_set): {len(dedup_set)}")

    skip_num = len(dedup_set)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = vllm.LLM(args.model_path, dtype="float16", tensor_parallel_size=4)

    def generate(text_list, temperature, max_tokens=32):
        # kwargs = {'top_k': 1} if temperature == 0.0 else {}
        kwargs = {}

        results = model.generate(
            text_list,
            vllm.SamplingParams(
                n=args.n,
                best_of=args.bestof,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["<eos>, <|eot_id|>", "</s>"],
                **kwargs,
            ),
        )

        return [r.outputs for r in results]

    def format_dialog(dialog):
        return tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )

    def make_batch(data):
        batches = []
        for il, c in enumerate(data):
            if il < skip_num:
                continue

            if len(batches) == args.batchsize:
                yield batches
                batches = []

            batches.append(c)

        if len(batches) != 0:
            yield batches

    for minibatch in tqdm(make_batch(dfdata)):
        to_generate = [
            format_dialog([{"role": "user", "content": bench["nmm_question"]}])
            for bench in minibatch
        ]

        results = generate(to_generate, args.temperature)

        for bench, res in zip(minibatch, results):
            llm_answer = res[0].text.strip()

            correct = check_llm_answer_correctness(
                llm_answer=llm_answer,
                expected_answer=bench["expected_answer"],
            )
            ans = {
                "nmm_question_id": bench["nmm_question_id"],
                "nmm_question_pair_id": bench["nmm_question_pair_id"],
                "wbk": bench["wbk"],
                "expected_answer": bench["expected_answer"],
                "llm_answer": llm_answer,
                "correct": correct,
            }

            df_ans = pd.DataFrame([ans])
            df_ans.to_csv(
                output_file_path,
                mode="a",
                header=not os.path.exists(output_file_path),
                index=False,
            )

    # Evaluate the results
    nmm1000_llm_results = pd.read_csv(output_file_path)
    evaluate_nmm1000_llm_results(nmm1000_llm_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str)

    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--logs_dir", type=str)
    parser.add_argument("--log_file_name", type=str)

    parser.add_argument("--batchsize", type=int, default=8)

    parser.add_argument(
        "--model_path",
        type=str,
    )

    parser.add_argument("--temperature", type=float)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--bestof", type=int, default=4)

    args = parser.parse_args()
    main(args)
