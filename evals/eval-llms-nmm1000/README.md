# Evaluating LLMs on Factual Questions about Natural Medicinal Materials

## Objective

This evaluation aims to assess how different Large Language Models (LLMs) perform in answering factual questions related to Natural Medicinal Materials (NMMs) under two conditions: (1) without any background knowledge provided, and (2) with relevant NMM background knowledge provided. This evaluation can reveal whether LLMs have already learned standardized NMM knowledge during their training and whether providing NMM-specific background information can improve their factual accuracy.

## Dataset

The dataset used for this evaluation is called NMM1000 (`nmm1000`), located in `./dataset/nmm1000`. It comprises 4,000 factual questions about 1,000 NMM entities, each presented in two versions: one without background knowledge and one with background knowledge. The questions are based on the standardized NMM knowledge defined in the [Systematic Nomenclature for Natural Medicinal Materials (SNNMM)](https://shennongalpha.westlake.edu.cn/doc/en/snnmm/) proposed by [ShennongAlpha](https://shennongalpha.westlake.edu.cn/). The dataset includes questions in both English and Chinese.

For example, consider the NMM with a generic name "Ma-huang" and NMM ID `nmm-0006`. We construct the following two questions:

```text
Q1 (without background knowledge):
What is the Natural Medicinal Material ID (NMM ID) of Ma-huang? Please only answer the NMM ID.

Q2 (with background knowledge):
What is the Natural Medicinal Material ID (NMM ID) of Ma-huang? Please only answer the NMM ID.
Background Knowledge:
NMM ID: nmm-0006
NMM Systematic Name: Ephedra equisetina vel intermedia vel sinica Stem-herbaceous
NMM Systematic Chinese Name: 木贼麻黄或中麻黄或草麻黄草质茎
NMM Generic Name: Ma-huang
NMM Generic Chinese Name: 麻黄
```

For both questions, the expected answer is `nmm-0006`.

However, for Q1, since models like `gpt-4o-2024-08-06` may not have learned standardized NMM knowledge, they might fail and produce an incorrect answer such as:

```text
The Natural Medicinal Material ID (NMM ID) of Ma-huang is NMM0001722.
```

The dataset is stored in `nmm1000-questions.csv`, which contains the following columns:

| Column | Type | Description | Examples |
| - | - | - | - |
| nmm_question_id | `str` | Unique identifier of the NMM question. A suffix of `_wobk` indicates no background knowledge; `_wbk` indicates background knowledge is provided. | `nmm_q_1_wobk`, `nmm_q_1_wbk` |
| nmm_question_pair_id | `str` | Unique identifier of the NMM question pair. For each `_wobk` question, there is a corresponding `_wbk` question with the same pair ID (e.g., `nmm_q_1_wobk` and `nmm_q_1_wbk` share `nmm_q_1`). | `nmm_q_1` |
| wbk | `bool` | Indicates whether the question includes background knowledge. `True` if `_wbk` is in the question ID, otherwise `False`. | `False`, `True` |
| type | `str` | The type of NMM fact the question is based on. `nmmsn` indicates NMM Systematic Name, `nmmsn_zh` indicates NMM Systematic Chinese Name, `nmmgn` indicates NMM Generic Name, `nmmgn_zh` indicates NMM Generic Chinese Name. | `nmmsn`, `nmmsn_zh`, `nmmgn`, `nmmgn_zh` |
| lang | `str` | The language of the question (`en` for English, `zh` for Chinese). | `en`, `zh` |
| nmm_question | `str` | The NMM question itself, with or without background knowledge. | `What is the Natural Medicinal Material ID (NMM ID) of Ma-huang? Please only answer the NMM ID.` |
| background_knowledge | `str` or empty | The background knowledge for the `_wbk` questions. For `_wobk` questions, this field is empty. | `Background Knowledge:\nNMM ID: nmm-0006\nNMM Systematic Name: Ephedra equisetina vel intermedia vel sinica Stem-herbaceous\nNMM Systematic Chinese Name: 木贼麻黄或中麻黄或草麻黄草质茎\nNMM Generic Name: Ma-huang\nNMM Generic Chinese Name: 麻黄` |
| expected_answer | `str` | The expected answer to the question. | `nmm-0006` |

## Evaluating LLMs

We evaluated the following LLMs:

- `gpt-4o-2024-08-06`
- `gpt-4o-mini-2024-07-18`
- `qwen-turbo-2024-09-19`
- `qwen2-7b`
- `llama-3.1-8b`

## Evaluation Code

The original evaluation code is available in the `./scripts/` directory.

## Evaluation Results

The evaluation results are stored in the `./results/` directory.

Since calling different LLMs’ APIs is time-consuming and potentially costly, we provide the evaluation results for direct inspection. Each model’s responses to the NMM1000 questions are saved in a file named `nmm1000-questions-results-<llm_name>.csv` (e.g., `nmm1000-questions-results-gpt-4o-2024-08-06.csv`). This CSV file contains the following columns:

| Column | Type | Description | Examples |
| - | - | - | - |
| nmm_question_id | `str` | Same as in `nmm1000.csv`. | `nmm_q_1_wobk` |
| nmm_question_pair_id | `str` | Same as in `nmm1000.csv`. | `nmm_q_1` |
| wbk | `bool` | Same as in `nmm1000.csv`. | `False` |
| expected_answer | `str` | Same as in `nmm1000.csv`. | `nmm-0006` |
| llm_answer | `str` | The model’s response to the `nmm_question`. | `The Natural Medicinal Material ID (NMM ID) of Ma-huang is NMM0001722.` |
| correct | `bool` | Whether the LLM’s answer matches the `expected_answer` (ignoring case). | `False` |

By analyzing these CSV files, we can compute the accuracy of each model on the NMM1000 dataset. The accuracy results are summarized in files named `nmm1000-questions-accuracy-<llm_name>.txt`, such as `nmm1000-questions-accuracy-gpt-4o-2024-08-06.txt`.

## Final Evaluation Results

| Model | Accuracy (without background knowledge) | Accuracy (with background knowledge) |
| - | - | - |
| gpt-4o-2024-08-06 | 0 | 0.99550 |
| gpt-4o-mini-2024-07-18 | 0 | 0.98275 |
| qwen-turbo-2024-09-19 | 0 | 0.99750 |
| qwen2-7b | 0 | 0.99825 |
| llama-3.1-8b | 0 | 0.99500 |

These results show that the models generally fail to answer NMM questions correctly without background knowledge. However, when the relevant NMM background knowledge is provided, their accuracy significantly improves, indicating that these LLMs can utilize the provided standardized NMM knowledge to give correct answers.
