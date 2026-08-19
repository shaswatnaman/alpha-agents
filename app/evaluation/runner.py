"""
Evaluation runner — runs experiments on the retrieval pipeline
and stores results in the database.

Usage:
    python -m app.evaluation.runner --experiment dense_only

Experiments:
    dense_only    — DenseRetriever only (baseline)
    lexical_only  — LexicalRetriever only
    hybrid        — Dense + Lexical + RRF (no MMR)
    hybrid_mmr    — Dense + Lexical + RRF + MMR (full pipeline)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import structlog

from app.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from app.repositories.database import get_db_context

log = structlog.get_logger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent.parent.parent / "eval_datasets" / "retrieval_eval.json"


async def run_retrieval_experiment(experiment_name: str) -> dict:
    """
    Run a retrieval experiment against the fixed eval dataset.

    The dataset format:
    [
      {
        "query": "...",
        "ticker": "AAPL",
        "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2", ...]
      },
      ...
    ]
    """
    if not EVAL_DATASET_PATH.exists():
        log.warning("eval_dataset_missing", path=str(EVAL_DATASET_PATH))
        return {"error": "eval_dataset_missing"}

    with open(EVAL_DATASET_PATH) as f:
        samples = json.load(f)

    k = 6   # retrieval_final_k
    all_precision, all_recall, all_ndcg = [], [], []

    async with get_db_context() as session:
        for sample in samples:
            query = sample["query"]
            ticker = sample["ticker"]
            relevant = sample["relevant_chunk_ids"]

            if experiment_name == "dense_only":
                from app.retrieval.retriever import DenseRetriever
                retriever = DenseRetriever(session)
                results = await retriever.retrieve(query, ticker, top_k=k)
                retrieved_ids = [r.chunk.id for r in results]

            elif experiment_name == "lexical_only":
                from app.retrieval.retriever import LexicalRetriever
                retriever = LexicalRetriever(session)
                results = await retriever.retrieve(query, ticker, top_k=k)
                retrieved_ids = [r.chunk.id for r in results]

            elif experiment_name == "hybrid":
                from app.retrieval.retriever import DenseRetriever, LexicalRetriever, reciprocal_rank_fusion
                dense = await DenseRetriever(session).retrieve(query, ticker, top_k=20)
                lexical = await LexicalRetriever(session).retrieve(query, ticker, top_k=20)
                fused = reciprocal_rank_fusion(dense, lexical)[:k]
                retrieved_ids = [r.chunk.id for r in fused]

            elif experiment_name == "hybrid_mmr":
                from app.retrieval.retriever import HybridRetriever
                h = HybridRetriever(session)
                evidence = await h.retrieve(query, ticker, final_k=k)
                retrieved_ids = [ev.chunk_id for ev in evidence]

            else:
                return {"error": f"Unknown experiment: {experiment_name}"}

            p = precision_at_k(retrieved_ids, relevant, k)
            r = recall_at_k(retrieved_ids, relevant, k)
            n = ndcg_at_k(retrieved_ids, relevant, k)
            all_precision.append(p)
            all_recall.append(r)
            all_ndcg.append(n)

    n_samples = len(samples)
    result = {
        "experiment": experiment_name,
        "n_samples": n_samples,
        "k": k,
        "mean_precision_at_k": sum(all_precision) / n_samples if n_samples else 0,
        "mean_recall_at_k": sum(all_recall) / n_samples if n_samples else 0,
        "mean_ndcg_at_k": sum(all_ndcg) / n_samples if n_samples else 0,
    }

    log.info("eval_result", **result)
    return result


def main() -> None:
    from app.observability.logging import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(description="AlphaAgents retrieval evaluation")
    parser.add_argument(
        "--experiment",
        choices=["dense_only", "lexical_only", "hybrid", "hybrid_mmr"],
        default="hybrid_mmr",
    )
    args = parser.parse_args()

    result = asyncio.run(run_retrieval_experiment(args.experiment))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
