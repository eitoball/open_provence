from __future__ import annotations

from open_provence.trainer import PruningTrainingArguments


def test_pruning_training_arguments_default_warmup_is_applied() -> None:
    """Confirm the warmup_steps (ratio) default actually reaches TrainingArguments.

    transformers v5 removed the `warmup_ratio` field, so `PruningTrainingArguments`
    must override `warmup_steps` directly; otherwise warmup is silently disabled
    (`get_warmup_steps` always returns 0).
    """
    args = PruningTrainingArguments(output_dir="/tmp/does-not-matter", bf16=False)

    num_training_steps = 1000
    warmup_steps = args.get_warmup_steps(num_training_steps)

    assert warmup_steps == 100  # 0.1 * 1000
