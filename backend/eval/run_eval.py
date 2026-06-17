"""Evaluate the Center Desk RAG system.

Two layers of metrics:

1. Retrieval (deterministic, free, always runs)
   - hit@1 / hit@3 : did a correct KB entry appear in the top-1 / top-3?
   - MRR          : mean reciprocal rank of the first correct entry (rewards
                    ranking the right doc higher).
   - guardrail    : on out-of-scope queries, does threshold filtering correctly
                    drop all context (so the assistant refuses)? And on in-scope
                    queries, does it avoid wrongly refusing?

2. Generation quality (optional, costs LLM calls) — pass --judge
   - faithfulness : is the generated answer grounded in the retrieved context?
   - correctness  : does it match the reference answer?
   Judged by the same free HF model (LLM-as-judge).

Run from the backend/ directory:
    python -m eval.run_eval                 # retrieval + guardrail only (free)
    python -m eval.run_eval --judge         # also LLM-judged generation
    python -m eval.run_eval --judge --limit 8
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.retriever import get_retriever
from eval.dataset import IN_SCOPE, OUT_OF_SCOPE

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- #
# Retrieval + guardrail metrics                                               #
# --------------------------------------------------------------------------- #
def evaluate_retrieval(k: int = 5) -> dict:
    retriever = get_retriever()

    # Sanity-check the eval set against the live KB so a typo in `expected`
    # doesn't silently look like a retrieval miss.
    kb_questions = {m["question"] for m in retriever._metadata}
    for item in IN_SCOPE:
        for exp in item["expected"]:
            if exp not in kb_questions:
                print(f"  [warn] expected question not in KB: {exp!r}")

    hits_at_1 = hits_at_3 = 0
    reciprocal_ranks = []
    false_refusals = 0
    per_item = []

    for item in IN_SCOPE:
        docs = retriever.search(item["query"], k=k)
        ranked_questions = [d.question for d in docs]
        expected = set(item["expected"])

        rank = next(
            (i + 1 for i, q in enumerate(ranked_questions) if q in expected), None
        )
        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # Would the live guardrail wrongly refuse this valid question?
        passed = retriever.search_filtered(item["query"])
        if not passed:
            false_refusals += 1

        per_item.append(
            {
                "query": item["query"],
                "rank": rank,
                "top_score": round(docs[0].score, 3) if docs else None,
            }
        )

    n = len(IN_SCOPE)

    # Out-of-scope: filtering should remove everything -> assistant refuses.
    correct_refusals = 0
    leaks = []
    for item in OUT_OF_SCOPE:
        passed = retriever.search_filtered(item["query"])
        if not passed:
            correct_refusals += 1
        else:
            leaks.append(
                {"query": item["query"], "top_score": round(passed[0].score, 3)}
            )

    m = len(OUT_OF_SCOPE)
    return {
        "n_in_scope": n,
        "hit@1": round(hits_at_1 / n, 3),
        "hit@3": round(hits_at_3 / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "false_refusal_rate": round(false_refusals / n, 3),
        "n_out_of_scope": m,
        "refusal_accuracy": round(correct_refusals / m, 3),
        "leaks": leaks,
        "per_item": per_item,
    }


# --------------------------------------------------------------------------- #
# Generation quality (LLM-as-judge)                                           #
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """You are grading a residence-hall front-desk assistant.

Question: {question}
Retrieved context:
{context}

Assistant answer: {answer}
Reference answer: {reference}

Grade two things, each strictly 1 (yes) or 0 (no):
- faithful: every claim in the assistant answer is supported by the retrieved context (no invented steps).
- correct: the assistant answer agrees with the reference answer.

Respond with ONLY a compact JSON object, e.g. {{"faithful": 1, "correct": 1}}."""


def _judge_one(client, model: str, question, context, answer, reference) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question, context=context, answer=answer, reference=reference
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"faithful": 0, "correct": 0, "raw": text}
    try:
        data = json.loads(match.group(0))
        return {"faithful": int(data.get("faithful", 0)), "correct": int(data.get("correct", 0))}
    except (json.JSONDecodeError, ValueError):
        return {"faithful": 0, "correct": 0, "raw": text}


def _build_judge(provider: str):
    """Return (client, model) for the judge. The generator is always the free
    HF model; the JUDGE is independent — preferably a stronger model so the
    grades are trustworthy (using the same weak model to generate and grade is
    an anti-pattern: weak discrimination + self-preference bias)."""
    if provider == "openai":
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set — add it to backend/.env or use --judge-provider hf")
        return OpenAI(api_key=settings.openai_api_key), settings.judge_model

    from huggingface_hub import InferenceClient

    if not settings.hf_token:
        raise RuntimeError("HF token not set — add HF_TOKEN to backend/.env")
    return InferenceClient(api_key=settings.hf_token), settings.hf_model


def evaluate_generation(provider: str = "openai", limit: int | None = None) -> dict:
    from app.rag import answer_text

    if not settings.hf_token:
        raise RuntimeError("HF token not set — the generator needs it. Add HF_TOKEN to backend/.env")

    client, judge_model = _build_judge(provider)
    retriever = get_retriever()
    items = IN_SCOPE[:limit] if limit else IN_SCOPE

    faithful = correct = errors = 0
    per_item = []
    for item in items:
        docs = retriever.search_filtered(item["query"])
        context = "\n\n".join(d.text for d in docs)
        # The free HF generator is occasionally flaky; one bad call shouldn't
        # sink the whole run. Retry once, then record the item as an error.
        answer = None
        for _ in range(2):
            try:
                answer = answer_text(item["query"])
                break
            except Exception as e:  # noqa: BLE001 — record and continue
                last_err = str(e)
        if not answer:
            errors += 1
            per_item.append({"query": item["query"], "error": last_err[:120]})
            continue

        grade = _judge_one(
            client, judge_model, item["query"], context, answer, item["reference"]
        )
        faithful += grade.get("faithful", 0)
        correct += grade.get("correct", 0)
        per_item.append({"query": item["query"], **grade})

    # Rates are over successfully-generated items so a flaky endpoint doesn't
    # silently deflate the quality score.
    n = len(items)
    scored = n - errors
    return {
        "n_items": n,
        "n_scored": scored,
        "n_errors": errors,
        "judge_provider": provider,
        "judge_model": judge_model,
        "generator_model": settings.hf_model,
        "faithfulness": round(faithful / scored, 3) if scored else None,
        "correctness": round(correct / scored, 3) if scored else None,
        "per_item": per_item,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Center Desk RAG system.")
    parser.add_argument("--judge", action="store_true", help="also run LLM-judged generation eval (uses the LLM)")
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "hf"],
        default="openai",
        help="which model grades the answers (default: openai — stronger, independent judge)",
    )
    parser.add_argument("--limit", type=int, default=None, help="limit number of generation items judged")
    args = parser.parse_args()

    started = time.time()
    print(f"Embedding model : {settings.embedding_model}")
    print(f"Score threshold : {settings.score_threshold}\n")

    print("== Retrieval & guardrail ==")
    retrieval = evaluate_retrieval()
    print(f"  in-scope items     : {retrieval['n_in_scope']}")
    print(f"  hit@1              : {retrieval['hit@1']}")
    print(f"  hit@3              : {retrieval['hit@3']}")
    print(f"  MRR                : {retrieval['mrr']}")
    print(f"  false-refusal rate : {retrieval['false_refusal_rate']}  (in-scope wrongly refused)")
    print(f"  refusal accuracy   : {retrieval['refusal_accuracy']}  (out-of-scope correctly refused)")
    if retrieval["leaks"]:
        print(f"  guardrail leaks    : {retrieval['leaks']}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model,
        "score_threshold": settings.score_threshold,
        "retrieval": retrieval,
    }

    if args.judge:
        print("\n== Generation quality (LLM-as-judge) ==")
        generation = evaluate_generation(provider=args.judge_provider, limit=args.limit)
        print(f"  generator     : {generation['generator_model']}")
        print(f"  judge         : {generation['judge_model']} ({generation['judge_provider']})")
        print(f"  items scored  : {generation['n_scored']}/{generation['n_items']}  (errors: {generation['n_errors']})")
        print(f"  faithfulness  : {generation['faithfulness']}")
        print(f"  correctness   : {generation['correctness']}")
        report["generation"] = generation

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
