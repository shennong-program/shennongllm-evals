from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

# Define paths
CURRENT_DIR = Path(__file__).resolve().parent
NMM1000E_DATASET_DIR = CURRENT_DIR / "../dataset/nmm1000e"
NMM1000E_QUERIES_CSV = NMM1000E_DATASET_DIR / "nmm1000-queries.csv"
RESULTS_DIR = CURRENT_DIR / "../results"

# Load queries data
DF_NMM_QUERIES = pd.read_csv(NMM1000E_QUERIES_CSV)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a (list[float]): First vector.
        b (list[float]): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_top_5_nmm_texts_by_nmm_query_embedding(
    df_nmm_texts_embedding: DataFrame,
    embedding: list[float],
) -> tuple[DataFrame, str, str, str, str]:
    """
    Search for the top 5 most similar nmm_texts based on cosine similarity.

    Args:
        df_nmm_texts_embedding (DataFrame): DataFrame containing nmm_text_id and embedding columns.
        embedding (list[float]): Query embedding to compare against.

    Returns:
        tuple[DataFrame, str, str, str, str]:
            1. DataFrame with top 5 matches.
            2. String of nmm_text_id:similarity pairs for top 5 matches.
            3. Comma-separated string of top 5 nmm_text_id.
            4. Comma-separated string of top 3 nmm_text_id.
            5. The nmm_text_id of the top match.
    """
    # Create a temporary DataFrame to avoid modifying the original
    temp_df = df_nmm_texts_embedding.copy()
    temp_df["similarity"] = temp_df.embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )
    res = temp_df.sort_values("similarity", ascending=False).head(5)

    # Create output strings
    top_5_with_similarity = ",".join(
        f"{row['nmm_text_id']}:{row['similarity']:.6f}" for _, row in res.iterrows()
    )
    top_5_ids = ",".join(res["nmm_text_id"].tolist())
    top_3_ids = ",".join(res["nmm_text_id"].tolist()[:3])
    top_1_id = res["nmm_text_id"].iloc[0]

    return res, top_5_with_similarity, top_5_ids, top_3_ids, top_1_id


def search_top_5_nmm_texts_on_df_nmm_queries_embedding(
    df_nmm_queries_embedding: DataFrame,
    df_nmm_texts_embedding: DataFrame,
    save_path: Path | None = None,
) -> DataFrame:
    """
    Search for top 5 nmm_texts for each query and return results in a DataFrame.
    Adds additional columns to indicate whether the expected ID is found in top N results.

    Args:
        df_nmm_queries_embedding (DataFrame): DataFrame containing nmm_query_id and embedding.
        df_nmm_texts_embedding (DataFrame): DataFrame containing nmm_text_id and embedding.
        save_path (Path | None): Path to save the output as CSV. If None, do not save.

    Returns:
        DataFrame: A DataFrame with the following columns:
            - nmm_query_id
            - expected_nmm_text_id
            - top_5_with_similarity
            - top_5_ids
            - top_3_ids
            - top_1_id
            - top_5_hit
            - top_3_hit
            - top_1_hit
    """
    # Ensure save_path directory exists if provided
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge DF_NMM_QUERIES to get expected_nmm_text_id
    df_nmm_queries_embedding = df_nmm_queries_embedding.merge(
        DF_NMM_QUERIES[["nmm_query_id", "expected_nmm_text_id"]],
        on="nmm_query_id",
        how="left",
    )

    # Define a helper function for row-wise computation
    def compute_top_n_columns(row):
        _, top_5_with_similarity, top_5_ids, top_3_ids, top_1_id = (
            search_top_5_nmm_texts_by_nmm_query_embedding(
                df_nmm_texts_embedding, row.embedding
            )
        )
        expected_id = row.expected_nmm_text_id
        return pd.Series(
            [
                top_5_with_similarity,
                top_5_ids,
                top_3_ids,
                top_1_id,
                expected_id in top_5_ids.split(","),
                expected_id in top_3_ids.split(","),
                expected_id == top_1_id,
            ],
            index=[
                "top_5_with_similarity",
                "top_5_ids",
                "top_3_ids",
                "top_1_id",
                "top_5_hit",
                "top_3_hit",
                "top_1_hit",
            ],
        )

    # Enable progress bar and apply the function row-wise
    tqdm.pandas(desc="Processing rows")
    df_nmm_queries_embedding = df_nmm_queries_embedding.join(
        df_nmm_queries_embedding.progress_apply(compute_top_n_columns, axis=1)  # type: ignore
    )

    # Select relevant columns for output
    columns_to_return = [
        "nmm_query_id",
        "expected_nmm_text_id",
        "top_5_with_similarity",
        "top_5_ids",
        "top_3_ids",
        "top_1_id",
        "top_5_hit",
        "top_3_hit",
        "top_1_hit",
    ]
    df_to_return = df_nmm_queries_embedding[columns_to_return]

    # Save to CSV if save_path is provided
    if save_path is not None:
        df_to_return.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    return df_to_return


def calculate_hit_ratio(df: DataFrame, hit_column: str) -> float:
    """
    Calculate the hit ratio for a specific hit column.

    Args:
        df (DataFrame): DataFrame containing the hit column.
        hit_column (str): Column name indicating hits (e.g., 'top_5_hit').

    Returns:
        float: Hit ratio as a decimal number between 0 and 1.
    """
    total_queries = len(df)
    total_hits = df[hit_column].sum()
    return total_hits / total_queries


def main():
    MODEL_NAMES = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]
    for model_name in MODEL_NAMES:
        print(f"Processing model: {model_name}")
        # Load embeddings
        df_nmm_texts_embedding = pd.read_parquet(
            RESULTS_DIR / f"nmm1000-texts-embedding-{model_name}.parquet"
        )
        df_nmm_queries_embedding = pd.read_parquet(
            RESULTS_DIR / f"nmm1000-queries-embedding-{model_name}.parquet"
        )

        # Search for top 5 nmm_texts for each query
        df_results = search_top_5_nmm_texts_on_df_nmm_queries_embedding(
            df_nmm_queries_embedding,
            df_nmm_texts_embedding,
            RESULTS_DIR / f"nmm1000-queries-top-5-text-embedding-{model_name}.csv",
        )

        hit_ratio_top_5 = calculate_hit_ratio(df_results, "top_5_hit")
        hit_ratio_top_3 = calculate_hit_ratio(df_results, "top_3_hit")
        hit_ratio_top_1 = calculate_hit_ratio(df_results, "top_1_hit")

        print(f"Hit Ratio@5: {hit_ratio_top_5}")
        print(f"Hit Ratio@3: {hit_ratio_top_3}")
        print(f"Hit Ratio@1: {hit_ratio_top_1}")

        # Save hit ratios to a file
        with open(
            RESULTS_DIR / f"nmm1000-queries-hit-ratio-{model_name}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Hit Ratio@5: {hit_ratio_top_5}\n")
            f.write(f"Hit Ratio@3: {hit_ratio_top_3}\n")
            f.write(f"Hit Ratio@1: {hit_ratio_top_1}\n")


if __name__ == "__main__":
    main()
