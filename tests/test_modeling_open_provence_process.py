from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest
from open_provence.modeling_open_provence_standalone import (
    OpenProvenceConfig,
    OpenProvenceModel,
    _FragmentRecord,
    english_sentence_splitter,
)

ENGLISH_MODEL_PATH = Path("output/open-provence-reranker-v1-gte-modernbert-base")
JAPANESE_MODEL_PATH = Path("output/open-provence-reranker-japanese-v20251021-bs256")
ENGLISH_RELEASE_MODEL_PATH = Path(
    "output/release_models/open-provence-reranker-v1-gte-modernbert-base"
)

ENGLISH_REMOTE_ID = "hotchpotch/open-provence-reranker-v1-gte-modernbert-base"
EN_JP_REMOTE_ID = "hotchpotch/open-provence-reranker-xsmall-v1"


def _requires_checkpoint(path: Path) -> bool:
    if not path.exists():
        # These integration tests are intended to run against real checkpoints.
        # The repository omits large weights, so missing artifacts are skipped here,
        # but CI/release environments should provide the checkpoints to exercise the
        # full process pipeline.
        pytest.skip(f"checkpoint directory {path} is missing")
        return False
    weights_file = path / "model.safetensors"
    if not weights_file.exists() or weights_file.stat().st_size < 1024 * 1024:
        pytest.skip(
            f"checkpoint {path} is a placeholder without full weights; skipping integration test"
        )
        return False
    return True


@cache
def _load_remote_model(model_id: str) -> OpenProvenceModel:
    config = OpenProvenceConfig.from_pretrained(model_id)
    model = OpenProvenceModel(config)
    model.eval()
    return model


@cache
def _tiny_bert_model() -> OpenProvenceModel:
    """Lightweight model used for shape/validation tests.

    Builds a single-layer BERT with a small hidden size to keep the unit tests fast
    while still exercising the full process() pipeline.
    """

    config = OpenProvenceConfig(
        base_model_config={
            "model_type": "bert",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "vocab_size": 30522,
        },
        tokenizer_name_or_path="bert-base-uncased",
        pruning_config={"hidden_size": 32},
        max_length=64,
    )
    model = OpenProvenceModel(config)
    model.eval()
    return model


def _load_tiny_model_or_skip() -> OpenProvenceModel:
    try:
        return _tiny_bert_model()
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"tiny test model unavailable: {exc}")


def _load_model_with_remote_fallback(
    checkpoint_dir: Path, remote_model_id: str
) -> OpenProvenceModel:
    weights_file = checkpoint_dir / "model.safetensors"
    if weights_file.exists() and weights_file.stat().st_size >= 1024 * 1024:
        return OpenProvenceModel.from_pretrained(str(checkpoint_dir))
    return _load_remote_model(remote_model_id)


def _build_single_fragment(model: OpenProvenceModel, document: str) -> list[_FragmentRecord]:
    token_ids = model.tokenizer.encode(document, add_special_tokens=False)
    return [
        _FragmentRecord(
            text=document,
            sentence_index=0,
            fragment_index=0,
            global_index=0,
            token_length=len(token_ids),
            token_ids=token_ids,
        )
    ]


@pytest.mark.parametrize(
    ("checkpoint", "remote_id", "question", "document"),
    [
        (
            ENGLISH_MODEL_PATH,
            ENGLISH_REMOTE_ID,
            "What is artificial intelligence?",
            "Artificial intelligence studies intelligent behaviour in machines.",
        ),
        (
            JAPANESE_MODEL_PATH,
            EN_JP_REMOTE_ID,
            "AIとは何ですか？",
            "AIは人工知能の略称で、人間の知能を機械で再現することを指します。",
        ),
    ],
)
def test_prepare_block_inputs_inserts_special_tokens(
    checkpoint: Path, remote_id: str, question: str, document: str
) -> None:
    model = _load_model_with_remote_fallback(checkpoint, remote_id)

    query_tokens = model.tokenizer.encode(question, add_special_tokens=False)
    fragments = _build_single_fragment(model, document)
    sep_token_ids = model.tokenizer.encode(
        model.tokenizer.sep_token or "", add_special_tokens=False
    )
    blocks = model._assemble_blocks_from_fragments(
        len(query_tokens), len(sep_token_ids), fragments
    )
    input_ids, attention_mask, token_type_ids, ranges = model._prepare_block_inputs(
        query_tokens, blocks[0]
    )

    assert input_ids, "input ids should not be empty"
    cls_candidates = [model.tokenizer.cls_token_id, model.tokenizer.bos_token_id]
    cls_candidates = [candidate for candidate in cls_candidates if isinstance(candidate, int)]
    if cls_candidates:
        assert input_ids[0] in cls_candidates, (
            f"expected CLS/BOS token at start, got {input_ids[0]} (candidates={cls_candidates})"
        )

    sep_candidates = [model.tokenizer.sep_token_id, model.tokenizer.eos_token_id]
    sep_candidates = [candidate for candidate in sep_candidates if isinstance(candidate, int)]
    if sep_candidates:
        assert any(token in sep_candidates for token in input_ids[1:]), (
            "no SEP/EOS token found in prepared inputs"
        )

    assert ranges, "context ranges must be populated"
    assert attention_mask == [1] * len(input_ids), "attention mask should align with inputs"
    if token_type_ids is not None:
        assert len(token_type_ids) == len(input_ids)


@pytest.mark.parametrize(
    ("checkpoint", "remote_id", "expected_flag"),
    [
        # Under transformers v5, build_inputs_with_special_tokens is gone from the standard
        # fast-tokenizer backend, so ModernBERT's tokenizer (which never overrode that
        # generic no-op method) now goes through the post_processor-template fallback --
        # which places CLS/SEP correctly, so the manual-override heuristic no longer
        # triggers here (it was True under v4, where the broken native method was used).
        (ENGLISH_MODEL_PATH, ENGLISH_REMOTE_ID, False),
        (JAPANESE_MODEL_PATH, EN_JP_REMOTE_ID, False),
    ],
)
def test_manual_special_token_detection(
    checkpoint: Path, remote_id: str, expected_flag: bool
) -> None:
    model = _load_model_with_remote_fallback(checkpoint, remote_id)
    actual_flag = getattr(model, "_manual_special_tokens_required", False)
    assert actual_flag is expected_flag


@pytest.mark.parametrize(
    ("checkpoint", "question", "base_text"),
    [
        (
            ENGLISH_MODEL_PATH,
            "What is artificial intelligence?",
            "Artificial intelligence allows machines to learn from data, adapt to new situations, and support people in solving complex problems across industry, science, and daily life. ",
        ),
        (
            JAPANESE_MODEL_PATH,
            "AIとは何ですか？",
            "人工知能は大量のデータから学習し、人間が扱いにくい複雑な課題に対して柔軟に適応し、日常生活や産業のさまざまな場面で意思決定を支援する技術です。",
        ),
    ],
)
def test_process_handles_long_document(checkpoint: Path, question: str, base_text: str) -> None:
    if not _requires_checkpoint(checkpoint):
        return

    model = OpenProvenceModel.from_pretrained(str(checkpoint))

    multiplier = (2000 // len(base_text)) + 2
    long_document = (base_text * multiplier)[:2000]

    result = model.process(
        question=question,
        context=[long_document],
        threshold=0.1,
        show_progress=False,
        return_sentence_metrics=True,
        return_sentence_texts=True,
    )

    assert "kept_sentences" in result
    assert "removed_sentences" in result
    kept_sentences = result.get("kept_sentences", [[]])
    assert kept_sentences, "process should return kept sentences for long document"
    pruned_contexts = result.get("pruned_context", [])
    assert len(pruned_contexts) == 1
    assert isinstance(pruned_contexts[0], str)


@pytest.mark.parametrize(
    ("checkpoint", "question", "base_text"),
    [
        (
            ENGLISH_MODEL_PATH,
            "What is artificial intelligence?",
            "Artificial intelligence allows machines to learn from data, adapt to new situations, and support people in solving complex problems across industry, science, and daily life. ",
        ),
        (
            JAPANESE_MODEL_PATH,
            "AIとは何ですか？",
            "人工知能は大量のデータから学習し、人間が扱いにくい複雑な課題に対して柔軟に適応し、日常生活や産業のさまざまな場面で意思決定を支援する技術です。",
        ),
    ],
)
def test_process_omits_sentence_texts_by_default(
    checkpoint: Path, question: str, base_text: str
) -> None:
    if not _requires_checkpoint(checkpoint):
        return

    model = OpenProvenceModel.from_pretrained(str(checkpoint))

    result = model.process(
        question=question,
        context=[base_text],
        threshold=0.1,
        show_progress=False,
    )

    assert "kept_sentences" not in result
    assert "removed_sentences" not in result


def test_english_sentence_splitter_preserves_newlines() -> None:
    context = (
        "Work deadlines piled up today, and I kept rambling about budget spreadsheets to my roommate.\n"
        "Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.\n"
        "Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.\n"
        "Later I remembered to water the plants and pay the electricity bill before finally getting some sleep.\n"
    )

    sentences = english_sentence_splitter(context)

    assert len(sentences) == 4
    assert sentences[0].endswith("\n")
    assert sentences[2].startswith("Sushi is honestly my favourite"), (
        "expected sushi sentence to be preserved"
    )


def test_process_filters_irrelevant_sentences_with_english_splitter() -> None:
    if not _requires_checkpoint(ENGLISH_RELEASE_MODEL_PATH):
        return

    model = OpenProvenceModel.from_pretrained(str(ENGLISH_RELEASE_MODEL_PATH))

    question = "What's your favorite Japanese food?"
    context = (
        "Work deadlines piled up today, and I kept rambling about budget spreadsheets to my roommate.\n"
        "Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.\n"
        "Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.\n"
        "Later I remembered to water the plants and pay the electricity bill before finally getting some sleep.\n"
    )

    result = model.process(
        question=question,
        context=context,
        threshold=0.1,
        show_progress=False,
        return_sentence_metrics=True,
        return_sentence_texts=True,
    )

    kept = result.get("kept_sentences", [])
    removed = result.get("removed_sentences", [])
    probs = result.get("sentence_probabilities", [])

    assert kept and kept[0].startswith("Sushi is honestly my favourite"), (
        "relevant sentence should be kept"
    )
    assert removed, "irrelevant sentences should be removed"
    assert all(sentence.endswith("\n") for sentence in removed), "splitter should retain newlines"
    assert probs, "sentence probabilities should be returned"
    assert probs[2] > 0.9
    assert probs[0] < 0.1 and probs[1] < 0.1 and probs[3] < 0.1


def test_process_accepts_aligned_question_and_context_lists() -> None:
    model = _load_tiny_model_or_skip()

    questions = ["What is sushi?", "How do you brew green tea?"]
    contexts = [
        "Sushi is vinegared rice paired with seafood or vegetables.",
        "Steep leaves in hot water just below boiling for a short time.",
    ]

    result = model.process(
        question=questions,
        context=contexts,
        threshold=0.1,
        show_progress=False,
    )

    pruned = result.get("pruned_context")
    scores = result.get("reranking_score")
    compression = result.get("compression_rate")

    assert isinstance(pruned, list)
    assert len(pruned) == len(questions)
    assert all(isinstance(item, str) for item in pruned)
    assert isinstance(scores, list)
    assert len(scores) == len(questions)
    assert isinstance(compression, list)
    assert len(compression) == len(questions)


def test_process_rejects_misaligned_question_context_lengths() -> None:
    model = _load_tiny_model_or_skip()

    with pytest.raises(ValueError):
        model.process(
            question=["What is sushi?", "How do you brew green tea?"],
            context=["Only one context provided."],
            show_progress=False,
        )


def test_process_handles_scalar_question_and_context_string() -> None:
    """question: str, context: str → outputs are scalars (single doc)."""

    model = _load_tiny_model_or_skip()

    question = "What is sushi?"
    context = "Sushi is vinegared rice paired with seafood."

    result = model.process(question=question, context=context, show_progress=False)

    assert isinstance(result.get("pruned_context"), str)
    assert isinstance(result.get("reranking_score"), (float, type(None)))
    assert isinstance(result.get("compression_rate"), float)


def test_process_treats_context_list_as_multiple_documents_for_single_query() -> None:
    """question: str, context: list[str] → one query with multiple documents."""

    model = _load_tiny_model_or_skip()

    question = "What is sushi?"
    contexts = [
        "Sushi is vinegared rice paired with seafood.",
        "It often comes with raw fish and seaweed.",
    ]

    result = model.process(question=question, context=contexts, show_progress=False)

    pruned = result.get("pruned_context")
    scores = result.get("reranking_score")
    compression = result.get("compression_rate")

    assert isinstance(pruned, list), "pruned_context should be a list of documents"
    assert len(pruned) == len(contexts)
    assert isinstance(scores, list) and len(scores) == len(contexts)
    assert isinstance(compression, list) and len(compression) == len(contexts)


def test_process_accepts_nested_pre_split_sentences_per_query() -> None:
    """question: list[str], context: list[list[str]] → per-query nested docs preserved."""

    model = _load_tiny_model_or_skip()

    questions = ["What is sushi?", "How do you brew green tea?"]
    sentence_list = [
        "Sushi is vinegared rice.",
        "It can include fish.",
    ]
    contexts = [sentence_list, sentence_list]

    result = model.process(question=questions, context=contexts, show_progress=False)

    pruned = result.get("pruned_context")
    scores = result.get("reranking_score")
    compression = result.get("compression_rate")

    assert isinstance(pruned, list) and len(pruned) == len(questions)
    assert all(isinstance(doc, list) for doc in pruned), "documents should stay nested per query"
    assert isinstance(scores, list) and len(scores) == len(questions)
    assert all(isinstance(s, list) for s in scores)
    assert isinstance(compression, list) and len(compression) == len(questions)
    assert all(isinstance(c, list) for c in compression)


def test_process_accepts_nested_sentences_for_single_query() -> None:
    """question: str, context: list[list[str]] → treated as one query / one doc (pre-split)."""

    model = _load_tiny_model_or_skip()

    question = "What is sushi?"
    sentence_list = ["Sushi is vinegared rice.", "It can include fish."]
    contexts = [sentence_list]

    result = model.process(question=question, context=contexts, show_progress=False)

    pruned = result.get("pruned_context")
    scores = result.get("reranking_score")
    compression = result.get("compression_rate")

    assert isinstance(pruned, list) and len(pruned) == 1
    assert isinstance(pruned[0], str)
    assert isinstance(scores, list) and len(scores) == 1 and not isinstance(scores[0], list)
    assert (
        isinstance(compression, list)
        and len(compression) == 1
        and not isinstance(compression[0], list)
    )
