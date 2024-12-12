# Evaluating Embedding Models on Vector Encoding and Retrieval Performance for Natural Medicinal Materials

## Objective

This test aims to evaluate the performance of various embedding models in encoding Natural Medicinal Material (NMM)-related texts and using these embeddings for vector search. Given a set of NMM-related queries and factual texts, we will assess whether embedding models can effectively encode these NMM queries/texts and identify related texts based on the cosine similarity between query and text embeddings.

## Dataset

We use the NMM1000E (`nmm1000e`) dataset, located in `./dataset/nmm1000e`. It contains two files:

- `nmm1000-texts.csv`: Contains 1,000 texts about 1,000 NMM entities.
- `nmm1000-queries.csv`: Contains 4,000 queries about NMM. Each query corresponds uniquely to one text in `nmm1000-texts.csv` that contains the factual information requested.

The NMM queries/texts are based on the standardized NMM knowledge defined by the [Systematic Nomenclature for Natural Medicinal Materials (SNNMM)](https://shennongalpha.westlake.edu.cn/doc/en/snnmm/) proposed by [ShennongAlpha](https://shennongalpha.westlake.edu.cn/). The dataset includes queries in both English and Chinese.

For example, consider an NMM entity with the Systematic Name "Ephedra equisetina vel intermedia vel sinica Stem-herbaceous" and NMM ID `nmm-0006`. An example query might be:

```text
Query:
What is the Natural Medicinal Material ID (NMM ID) of Ephedra equisetina vel intermedia vel sinica Stem-herbaceous?
```

The goal is to find in `nmm1000-texts.csv` the text that contains the factual information about Ephedra equisetina vel intermedia vel sinica Stem-herbaceous.” For instance, the corresponding text is:

```text
Text:
NMM ID: nmm-0006. NMM Systematic Name: Ephedra equisetina vel intermedia vel sinica Stem-herbaceous. NMM Systematic Chinese Name: 木贼麻黄或中麻黄或草麻黄草质茎. NMM Generic Name: Ma-huang. NMM Generic Chinese Name: 麻黄.
```

### `nmm1000-texts.csv` Format

| Column | Type | Description | Example |
| - | - | - | - |
| `nmm_text_index` | `int` | Sequential index of the NMM text (1 to 1000). | `6` |
| `nmm_text_id`    | `str` | Unique ID for the NMM text. The format is `nmm-text-<nmm-id>`, where `<nmm-id>` is the NMM entity’s ID. | `nmm-text-nmm-0006` |
| `nmm_text` | `str` | The factual text about the NMM entity. | `NMM ID: nmm-0006. NMM Systematic Name: Ephedra equisetina vel intermedia vel sinica Stem-herbaceous. NMM Systematic Chinese Name: 木贼麻黄或中麻黄或草麻黄草质茎. NMM Generic Name: Ma-huang. NMM Generic Chinese Name: 麻黄.` |

### `nmm1000-queries.csv` Format

| Column | Type | Description | Example |
| - | - | - | - |
| `nmm_query_id` | `str`| Unique identifier for the NMM query. | `nmm-query-21` |
| `type` | `str` | The type of NMM fact the query is about (`nmmsn`, `nmmsn_zh`, `nmmgn`, `nmmgn_zh`). | `nmmsn` |
| `lang` | `str` | The language of the query (`en` for English, `zh` for Chinese). | `en` |
| `nmm_query` | `str`| The NMM query text. | `What is the Natural Medicinal Material ID (NMM ID) of Ephedra equisetina vel intermedia vel sinica Stem-herbaceous?` |
| `expected_nmm_text_id` | `str` | The unique NMM text ID that answers the query. | `nmm-text-nmm-0006` |

## Evaluated Embedding Models

We evaluated the following embedding models:

- `text-embedding-ada-002`
- `text-embedding-3-small`
- `text-embedding-3-large`

Each model encodes text into vectors of different dimensions (1536, 1536, and 3072, respectively). As a result, the `text-embedding-3-large` model requires twice as much storage space and additional computational cost for similarity calculations compared to the other two models.

## Evaluation Code

The original evaluation code is located in the `./scripts/` directory.

## Evaluation Results

### Embedding NMM Queries and Texts

The encoded embeddings of NMM queries and texts for each model are stored in the `./results/` directory in Parquet format, named `nmm1000-queries-embedding-<model-name>.parquet` and `nmm1000-texts-embedding-<model-name>.parquet`.

Parquet is chosen for efficient columnar storage and compression, reducing storage space for large amounts of embedding vectors.

You can load these embeddings using `pandas`:

```python
import pandas as pd

# Load NMM queries embeddings
nmm_queries_embedding = pd.read_parquet("nmm1000-queries-embedding-text-embedding-ada-002.parquet")

# Load NMM texts embeddings
nmm_texts_embedding = pd.read_parquet("nmm1000-texts-embedding-text-embedding-ada-002.parquet")
```

Each `pd.DataFrame` contains:

| Column | Type | Description |
| - | - | - |
| `nmm_query_id` / `nmm_text_id` | `str` | The unique ID of the query/text. |
| `embedding` | `list[float]` | The embedding vector of the query/text. Dimensions depend on the model. |

### Evaluating Query-Text Similarity

We use cosine similarity to evaluate the similarity between query and text embeddings. After computing cosine similarity, we rank texts by similarity and measure search accuracy and performance.

Evaluation results for each embedding model are stored in `./results/` as `nmm1000-queries-top-5-text-embedding-<model-name>.csv`.

For `text-embedding-ada-002`, this CSV includes:

| Column | Type | Description | Example |
| - | - | - | - |
| `nmm_query_id`| `str` | Unique NMM query ID | `nmm-query-21` |
| `expected_nmm_text_id`| `str` | The unique NMM text ID that should answer the query | `nmm-text-nmm-0006` |
| `top_5_with_similarity` | `str` | The top 5 most similar NMM texts and their similarities, formatted as `<nmm-text-id>:<similarity>,...` | `nmm-text-nmm-0006:0.924966,nmm-text-nmm-0004:0.921083,nmm-text-nmm-000a:0.920716,nmm-text-nmm-0008:0.917368,nmm-text-nmm-000e:0.917109` |
| `top_5_ids` | `str` | The top 5 most similar NMM text IDs in descending order of similarity, comma-separated | `nmm-text-nmm-0006,nmm-text-nmm-0004,nmm-text-nmm-000a,nmm-text-nmm-0008,nmm-text-nmm-000e`        |
| `top_3_ids` | `str` | The top 3 most similar NMM text IDs, comma-separated | `nmm-text-nmm-0006,nmm-text-nmm-0004,nmm-text-nmm-000a` |
| `top_1_id` | `str` | The single most similar NMM text ID | `nmm-text-nmm-0006` |
| `top_5_hit` | `bool` | Whether the expected text is among the top 5 results | `True` |
| `top_3_hit` | `bool` | Whether the expected text is among the top 3 results | `True` |
| `top_1_hit` | `bool` | Whether the expected text is the top 1 result | `True` |

### Calculating Hit Ratio @ N

We computed the Hit Ratio @ N at N=5, 3, and 1 for the 4,000 queries. The Hit Ratio @ N is the fraction of queries for which the target text appears in the top N retrieved texts. For example, if 3,000 out of 4,000 queries have the target text in the top 5 results, then the Hit Ratio @ 5 is 0.75 (or 75%).

By analyzing `nmm1000-queries-top-5-text-embedding-<model-name>.csv`, we obtain Hit Ratios @ N for each embedding model. The results are stored in `./results/`, for example in `nmm1000-queries-hit-ratio-embedding-<model-name>.csv`.

| Model | Hit Ratio @ 5 | Hit Ratio @ 3 | Hit Ratio @ 1 |
| - | - | - | - |
| `text-embedding-ada-002` | 0.63300 | 0.57550 | 0.41150 |
| `text-embedding-3-small` | 0.76475 | 0.69600 | 0.46575 |
| `text-embedding-3-large` | 0.80600 | 0.72950 | 0.46725 |

These results indicate that `text-embedding-3-large` generally provides the best performance among the three models tested, followed by `text-embedding-3-small`, and finally `text-embedding-ada-002`.
