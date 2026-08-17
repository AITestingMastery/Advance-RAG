"""
End-to-end evaluation for the Advanced RAG app.

Evaluates the real advanced_rag.py pipeline in two complementary ways:

1. ROUTER / STRUCTURED EVALUATION
   - Router accuracy
   - Structured filter correctness
   - Exact lookup correctness
   - Aggregation/count correctness
   - Numeric filtering correctness

2. HYBRID RAG + RAGAS EVALUATION
   - Uses real Hybrid Retrieval (BM25 + vector + RRF)
   - Uses real answer generation
   - Evaluates with RAGAS:
       * Faithfulness
       * Answer Relevancy
       * Context Precision
       * Context Recall

IMPORTANT:
The supplied Employee Master Data PDF is primarily a structured employee
dataset. Therefore the Hybrid/RAGAS questions are document-level semantic
questions rather than artificial employee filters.

For those RAGAS questions:
- First, the real router is tested normally.
- Then the question is independently evaluated through the real Hybrid
  Retrieval path so RAGAS can measure the Hybrid RAG capability even if the
  router correctly decides that a structured path is better.

This keeps router evaluation and RAGAS evaluation independent.

RUN:

    python3 evaluate_rag.py --out results/eval_report.csv

Router/filter/count evaluation only:

    python3 evaluate_rag.py --skip-ragas

Requires:

    ragas
    langchain-community==0.3.0
    datasets
    pandas
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import advanced_rag as rag  # noqa: E402

# We deliberately use ragas's legacy Langchain*Wrapper classes (see run_ragas()
# below) because the modern default resolution is broken for this app's
# needs -- embeddings lack embed_query, and the LLM adapter silently returns
# 1 generation instead of the 3 answer_relevancy asks for. Both wrappers warn
# DeprecationWarning at the CALLER's stack frame (stacklevel=2), not ragas's
# own module, so a module="ragas" filter can't catch them -- ignoring the
# category outright is the only thing that works here. This script has no
# other third-party deprecation risk, so this only silences the ones we're
# knowingly triggering; real warnings (network errors, retry notices, etc.)
# are logger.warning() calls, unaffected by this filter.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================================
# Strategy mapping
# ============================================================================

STRATEGY_LABEL_TO_KEY = {
    "Structured filtering": "structured_filter",
    "Structured aggregation": "aggregation",
    "Exact structured lookup": "exact_lookup",
    "Hybrid retrieval": "hybrid_retrieval",
}


# ============================================================================
# Ground truth
#
# Structured questions use hand-written oracle filters.
#
# Hybrid/RAGAS questions use hand-written reference answers because they are
# semantic/document-level questions rather than exact record filters.
# ============================================================================

GROUND_TRUTH = [

    # ------------------------------------------------------------------------
    # Structured filtering
    # ------------------------------------------------------------------------

    {
        "id": "Q1",
        "question": "List all employees in the Engineering department",
        "expected_strategy": "structured_filter",
        "oracle_filters": [
            {
                "field": "department",
                "operator": "equals",
                "value": "Engineering",
            }
        ],
    },

    {
        "id": "Q2",
        "question": "List all employees in the Engineering department who are on leave.",
        "expected_strategy": "structured_filter",
        "oracle_filters": [
            {
                "field": "department",
                "operator": "equals",
                "value": "Engineering",
            },
            {
                "field": "status",
                "operator": "equals",
                "value": "On Leave",
            },
        ],
    },

    {
        "id": "Q3",
        "question": 'Which employees work in Bengaluru and have the status "Active"?',
        "expected_strategy": "structured_filter",
        "oracle_filters": [
            {
                "field": "location",
                "operator": "equals",
                "value": "Bengaluru",
            },
            {
                "field": "status",
                "operator": "equals",
                "value": "Active",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------------

    {
        "id": "Q4",
        "question": 'How many employees are currently "On Leave"?',
        "expected_strategy": "aggregation",
        "oracle_filters": [
            {
                "field": "status",
                "operator": "equals",
                "value": "On Leave",
            }
        ],
    },

    # ------------------------------------------------------------------------
    # Exact lookup
    # ------------------------------------------------------------------------

    {
        "id": "Q5",
        "question": "Show the employee record for EMP057.",
        "expected_strategy": "exact_lookup",
        "oracle_filters": [
            {
                "field": "emp_id",
                "operator": "equals",
                "value": "EMP057",
            }
        ],
    },

    # ------------------------------------------------------------------------
    # Extended structured tests
    # ------------------------------------------------------------------------

    {
        "id": "EXT-negative",
        "question": "List employees in the Legal department based in Antarctica.",
        "expected_strategy": "structured_filter",
        "oracle_filters": [
            {
                "field": "department",
                "operator": "equals",
                "value": "Legal",
            },
            {
                "field": "location",
                "operator": "equals",
                "value": "Antarctica",
            },
        ],
        "expect_zero": True,
    },

    {
        "id": "EXT-or",
        "question": "List employees in Finance or IT department.",
        "expected_strategy": "structured_filter",
        "oracle_filters": [
            {
                "field": "department",
                "operator": "in",
                "value": ["Finance", "IT"],
            }
        ],
    },

    {
        "id": "EXT-numeric",
        "question": "How many employees have a salary above 60000?",
        "expected_strategy": "aggregation",
        "oracle_filters": [
            {
                "field": "salary_inr",
                "operator": "gt",
                "value": "60000",
            }
        ],
    },

    # ------------------------------------------------------------------------
    # HYBRID / RAGAS QUESTIONS
    #
    # These are deliberately semantic/document-level questions.
    #
    # The router is evaluated normally first.
    #
    # For RAGAS, the Hybrid path is independently exercised so that RAGAS
    # actually evaluates your BM25 + vector + RRF + generation pipeline.
    # ------------------------------------------------------------------------

    {
        "id": "RAGAS-1",
        "question": (
            "According to the Employee Master Data document, "
            "what information is recorded for each employee?"
        ),
        "expected_strategy": "hybrid_retrieval",
        "reference_answer": (
            "The Employee Master Data records the employee ID, name, "
            "department, position, email, phone number, hire date, salary "
            "in INR, location, manager, and employment status for each "
            "employee."
        ),
    },

    {
        "id": "RAGAS-2",
        "question": (
            "Give a high-level overview of the kinds of departments and "
            "job roles represented in the Employee Master Data."
        ),
        "expected_strategy": "hybrid_retrieval",
        "reference_answer": (
            "The dataset represents departments including Engineering, "
            "Information Technology, Finance, Human Resources, Sales, "
            "Marketing, Product, Operations, Legal, and Customer Support. "
            "Examples of roles include QA Engineer, Software Engineer, "
            "Senior Software Engineer, Engineering Manager, IT Support "
            "Specialist, System Administrator, Network Engineer, IT Manager, "
            "Accountant, Payroll Specialist, Finance Manager, HR Executive, "
            "Recruiter, Sales Manager, Product Manager, UX Designer, "
            "Operations Executive, Legal Associate, and Customer Success "
            "Manager."
        ),
    },

    {
        "id": "RAGAS-3",
        "question": (
            "Compare the types of technical roles represented in the "
            "Engineering and IT teams in the employee dataset."
        ),
        "expected_strategy": "hybrid_retrieval",
        "reference_answer": (
            "Engineering includes QA Engineer, Software Engineer, Senior "
            "Software Engineer, and Engineering Manager roles. IT includes "
            "IT Support Specialist, System Administrator, Network Engineer, "
            "and IT Manager roles. The Engineering records therefore focus "
            "more on software development, quality assurance, and engineering "
            "management, while the IT records cover support, systems, "
            "networking, and IT management."
        ),
    },

    {
        "id": "RAGAS-4",
        "question": (
            "What fields and categories are used to describe employees "
            "throughout the Employee Master Data document?"
        ),
        "expected_strategy": "hybrid_retrieval",
        "reference_answer": (
            "Each employee is described using identifying information such "
            "as employee ID and name, organizational information such as "
            "department, position, and manager, contact information such as "
            "email and phone, employment information such as hire date and "
            "status, and compensation and location information such as "
            "salary in INR and location."
        ),
    },
]


# ============================================================================
# Data helpers
# ============================================================================

def load_all_records():
    """
    Load all structured records already indexed by advanced_rag.py.
    """

    docs = rag.list_documents()

    if not docs:
        sys.exit(
            "No documents indexed in storage/rag.db. "
            "Upload the Employee Master Data using "
            "`streamlit run app.py` first, or call "
            "`advanced_rag.process_document()` once."
        )

    records = []

    for document in docs:
        records.extend(
            rag.get_records(document["document_id"])
        )

    return records


def oracle_count(records, filters):
    """
    Compute expected structured record count using hand-specified filters.

    IMPORTANT:
    The filters are NOT generated by the LLM router.
    """

    if not filters:
        return None

    return len(
        rag.apply_filters(records, filters)
    )


def extract_number(text):
    """
    Extract the first integer from an answer.
    Used only for aggregation sanity checks.
    """

    match = re.search(r"\d+", str(text))

    return int(match.group(0)) if match else None


# ============================================================================
# RAGAS
# ============================================================================

def run_ragas(records):
    """
    Run RAGAS against Hybrid RAG results.

    Each record must contain:

        question
        answer
        contexts
        reference_answer

    Returns:
        {
            "faithfulness": ...,
            "answer_relevancy": ...,
            "context_precision": ...,
            "context_recall": ...
        }

    Plus per-question scores.
    """

    if not records:
        return None, []

    try:
        from datasets import Dataset

        from ragas import evaluate

        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        # answer_relevancy calls self.embeddings.embed_query(...) internally
        # (ragas/metrics/_answer_relevance.py) -- a method that only exists on
        # ragas's OLD LangchainEmbeddingsWrapper. Left to its own defaults,
        # ragas 0.4.3's evaluate() resolves embeddings via its NEW-style
        # ragas.embeddings.openai_provider.OpenAIEmbeddings (a BaseRagasEmbedding,
        # not a langchain wrapper), which has no embed_query method at all --
        # that's the exact "'OpenAIEmbeddings' object has no attribute
        # 'embed_query'" AttributeError from the last run, and why
        # answer_relevancy alone came back nan while faithfulness/context_*
        # (which don't touch embeddings the same way) scored fine. Building
        # the legacy wrapper explicitly and passing it to evaluate() forces
        # the code path answer_relevancy actually expects.
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai.embeddings import OpenAIEmbeddings as _LCOpenAIEmbeddings

        _ragas_embeddings = LangchainEmbeddingsWrapper(
            _LCOpenAIEmbeddings(model=rag.EMBED_MODEL)
        )

        # Second instance of the same underlying problem: with no explicit
        # llm= passed, evaluate() auto-resolves its LLM through
        # ragas.llms.llm_factory(), which returns an InstructorBaseRagasLLM
        # (the "instructor" library's structured-output adapter). That
        # adapter doesn't support requesting multiple parallel completions
        # (n=3, used by answer_relevancy's strictness setting to average 3
        # reverse-engineered questions) -- it always returns exactly 1
        # generation regardless of what n was asked for. That's the
        # "LLM returned 1 generations instead of requested 3" warning firing
        # on every single question, and likely a chunk of the ~47s/item
        # runtime (each call goes through the instructor library's
        # validation/retry loop for a single structured JSON object instead
        # of one plain completion).
        #
        # LangchainLLMWrapper wraps a real langchain ChatOpenAI, which either
        # requests n completions natively in one call (when supported) or
        # falls back to n sequential calls -- either way it reliably returns
        # exactly n generations (see is_multiple_completion_supported() in
        # ragas/llms/base.py), which is what answer_relevancy actually needs.
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI as _LCChatOpenAI

        _ragas_llm = LangchainLLMWrapper(
            _LCChatOpenAI(model=rag.CHAT_MODEL, temperature=0)
        )

    except ImportError:
        print(
            "\nRAGAS is not installed correctly.\n"
            "Install with:\n\n"
            'uv pip install ragas "langchain-community==0.3.0" '
            "datasets pandas\n",
            file=sys.stderr,
        )

        return None, []

    dataset_rows = []

    for record in records:

        dataset_rows.append(
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"] or [""],
                "ground_truth": record["reference_answer"],
            }
        )

    dataset = Dataset.from_list(dataset_rows)

    print(
        f"\nRunning RAGAS on {len(dataset_rows)} Hybrid RAG question(s)..."
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        embeddings=_ragas_embeddings,
        llm=_ragas_llm,
    )

    dataframe = result.to_pandas()

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    # ------------------------------------------------------------
    # Per-question RAGAS scores
    # ------------------------------------------------------------

    per_question_scores = []

    for index, record in enumerate(records):

        scores = {
            "id": record["id"],
        }

        for metric in metric_names:

            if metric in dataframe.columns:

                value = dataframe.iloc[index][metric]

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None

                scores[metric] = value

            else:
                scores[metric] = None

        per_question_scores.append(scores)

    # ------------------------------------------------------------
    # Average scores
    # ------------------------------------------------------------

    averages = {}

    for metric in metric_names:

        if metric in dataframe.columns:

            numeric_values = []

            for value in dataframe[metric]:

                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    pass

            if numeric_values:
                averages[metric] = (
                    sum(numeric_values) / len(numeric_values)
                )

    return averages, per_question_scores


# ============================================================================
# Force Hybrid evaluation
# ============================================================================

def run_forced_hybrid_question(question, top_k):
    """
    Run a question through the REAL Hybrid Retrieval + generation path.

    Why force Hybrid?

    The Employee Master Data PDF is almost entirely structured data.
    The real router may correctly decide that a semantic question can be
    handled using a structured strategy.

    That is good router behavior, but it would prevent us from evaluating
    the Hybrid RAG engine with RAGAS.

    Therefore:
        - Normal query_document() tests the router.
        - This function temporarily forces the strategy to hybrid_retrieval.
        - All retrieval/generation code remains real.
        - Only the router decision is bypassed for this isolated benchmark.
    """

    original_classify_query = rag.classify_query

    def forced_hybrid_classifier(question_text, records):
        return {
            "strategy": "hybrid_retrieval",
            "operation": None,
            "filters": [],
            "reason": "Forced Hybrid benchmark for RAGAS evaluation",
        }

    try:

        rag.classify_query = forced_hybrid_classifier

        result = rag.query_document(
            question,
            document_id=None,
            top_k=top_k,
        )

    finally:

        rag.classify_query = original_classify_query

    return result


# ============================================================================
# Main evaluation
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Advanced RAG router, structured paths and Hybrid RAG with RAGAS."
    )

    parser.add_argument(
        "--out",
        default="results/eval_report.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help=(
            "Run router/filter/count evaluation only. "
            "Skip RAGAS and forced Hybrid benchmark."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=rag.TOP_K,
        help="Number of chunks/records used by retrieval.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Load indexed data
    # ------------------------------------------------------------------------

    all_records = load_all_records()

    print(
        f"Loaded {len(all_records)} indexed records across "
        f"{len(rag.list_documents())} document(s)."
    )

    per_question = []

    ragas_input = []

    # ------------------------------------------------------------------------
    # NORMAL ROUTER EVALUATION
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ROUTER / STRUCTURED EVALUATION")
    print("=" * 80)

    for ground_truth in GROUND_TRUTH:

        question = ground_truth["question"]

        # ------------------------------------------------------------
        # Run REAL application pipeline
        # ------------------------------------------------------------

        result = rag.query_document(
            question,
            document_id=None,
            top_k=args.top_k,
        )

        actual_strategy = STRATEGY_LABEL_TO_KEY.get(
            result["strategy"],
            "unknown",
        )

        expected_strategy = ground_truth["expected_strategy"]

        router_correct = (
            actual_strategy == expected_strategy
        )

        # ------------------------------------------------------------
        # Structured expected count
        # ------------------------------------------------------------

        expected_count = None

        if "oracle_filters" in ground_truth:

            expected_count = oracle_count(
                all_records,
                ground_truth["oracle_filters"],
            )

            if ground_truth.get("expect_zero"):
                expected_count = 0

        # ------------------------------------------------------------
        # Count correctness
        # ------------------------------------------------------------

        count_correct = None

        if (
            expected_count is not None
            and actual_strategy
            in (
                "structured_filter",
                "aggregation",
                "exact_lookup",
            )
        ):

            actual_count = result.get(
                "record_count",
                0,
            )

            count_correct = (
                actual_count == expected_count
            )

        # ------------------------------------------------------------
        # Aggregation answer correctness
        # ------------------------------------------------------------

        answer_number = extract_number(
            result.get("answer", "")
        )

        answer_matches_expected_number = None

        if actual_strategy == "aggregation":

            answer_matches_expected_number = (
                answer_number == expected_count
            )

        # ------------------------------------------------------------
        # Save row
        # ------------------------------------------------------------

        row = {
            "id": ground_truth["id"],
            "evaluation_type": (
                "structured"
                if "oracle_filters" in ground_truth
                else "hybrid_ragas"
            ),
            "question": question,
            "expected_strategy": expected_strategy,
            "actual_strategy": actual_strategy,
            "router_correct": router_correct,
            "expected_count": expected_count,
            "actual_record_count": result.get(
                "record_count"
            ),
            "count_correct": count_correct,
            "answer_matches_expected_number": (
                answer_matches_expected_number
            ),
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_context_precision": None,
            "ragas_context_recall": None,
            "answer": result.get(
                "answer",
                "",
            ),
        }

        per_question.append(row)

        # ------------------------------------------------------------
        # Console output
        # ------------------------------------------------------------

        router_status = (
            "OK"
            if router_correct
            else f"WRONG ({actual_strategy})"
        )

        message = (
            f"[{ground_truth['id']}] "
            f"router={router_status}"
        )

        if count_correct is not None:

            message += (
                f", count expected={expected_count} "
                f"actual={result.get('record_count')}"
            )

        print(message)

        # ------------------------------------------------------------
        # If this is a Hybrid/RAGAS test, prepare the separate
        # forced-Hybrid evaluation.
        # ------------------------------------------------------------

        if (
            expected_strategy == "hybrid_retrieval"
            and not args.skip_ragas
        ):

            print(
                f"  -> Running isolated Hybrid RAG path for "
                f"{ground_truth['id']}..."
            )

            hybrid_result = run_forced_hybrid_question(
                question,
                args.top_k,
            )

            hybrid_actual_strategy = STRATEGY_LABEL_TO_KEY.get(
                hybrid_result["strategy"],
                "unknown",
            )

            print(
                f"  -> Hybrid strategy: {hybrid_actual_strategy}"
            )

            contexts = [
                source.get("text", "")
                for source in hybrid_result.get(
                    "sources",
                    [],
                )
                if source.get("text")
            ]

            ragas_input.append(
                {
                    "id": ground_truth["id"],
                    "question": question,
                    "answer": hybrid_result.get(
                        "answer",
                        "",
                    ),
                    "contexts": contexts,
                    "reference_answer": ground_truth[
                        "reference_answer"
                    ],
                }
            )

            # Store the Hybrid answer in the row as well.
            row["hybrid_actual_strategy"] = (
                hybrid_actual_strategy
            )

            row["hybrid_answer"] = hybrid_result.get(
                "answer",
                "",
            )

            row["hybrid_context_count"] = len(
                contexts
            )

    # =========================================================================
    # Router statistics
    # =========================================================================

    total_questions = len(per_question)

    router_accuracy = (
        sum(
            bool(row["router_correct"])
            for row in per_question
        )
        / total_questions
        if total_questions
        else 0
    )

    structured_rows = [
        row
        for row in per_question
        if row["count_correct"] is not None
    ]

    count_accuracy = None

    if structured_rows:

        count_accuracy = (
            sum(
                bool(row["count_correct"])
                for row in structured_rows
            )
            / len(structured_rows)
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Router accuracy: "
        f"{router_accuracy:.0%} "
        f"({total_questions} questions)"
    )

    if count_accuracy is not None:

        print(
            f"Filter/count accuracy: "
            f"{count_accuracy:.0%} "
            f"({len(structured_rows)} structured questions)"
        )

    # =========================================================================
    # RAGAS
    # =========================================================================

    if not args.skip_ragas and ragas_input:

        print("\n" + "=" * 80)
        print("RAGAS HYBRID RAG EVALUATION")
        print("=" * 80)

        ragas_averages, ragas_per_question = run_ragas(
            ragas_input
        )

        if ragas_averages:

            print(
                "\nRAGAS average scores:"
            )

            for metric, value in ragas_averages.items():

                print(
                    f"  {metric}: "
                    f"{value:.3f}"
                )

            # ------------------------------------------------------------
            # Merge per-question RAGAS scores into report
            # ------------------------------------------------------------

            scores_by_id = {
                score["id"]: score
                for score in ragas_per_question
            }

            for row in per_question:

                score = scores_by_id.get(
                    row["id"]
                )

                if not score:
                    continue

                row["ragas_faithfulness"] = score.get(
                    "faithfulness"
                )

                row["ragas_answer_relevancy"] = score.get(
                    "answer_relevancy"
                )

                row["ragas_context_precision"] = score.get(
                    "context_precision"
                )

                row["ragas_context_recall"] = score.get(
                    "context_recall"
                )

        else:

            print(
                "RAGAS did not return any scores."
            )

    elif not args.skip_ragas:

        print(
            "\nNo RAGAS questions were configured."
        )

    elif args.skip_ragas:

        print(
            "\nRAGAS skipped because "
            "--skip-ragas was supplied."
        )

    # =========================================================================
    # Save CSV
    # =========================================================================

    output_path = Path(args.out)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        import pandas as pd

        dataframe = pd.DataFrame(
            per_question
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\nSaved evaluation report to "
            f"{output_path}"
        )

    except ImportError:

        json_path = output_path.with_suffix(
            ".json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                per_question,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\npandas unavailable. "
            f"Saved JSON report to {json_path}"
        )


if __name__ == "__main__":
    main()