"""Every experiment file must load and refer to rewards that exist."""

from pathlib import Path

import pytest

from open_r1.grounded.config import apply_overrides, load_config, reward_spec
from open_r1.grounded.rewards import RewardSettings, build_reward_functions

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "grounded"
ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
GRPO_CONFIGS = [p for p in ALL_CONFIGS if p.name.startswith(("grpo_", "ablation_"))]
MAIN = CONFIG_DIR / "grpo_qwen0.5b.yaml"


def _without(config, *dotted_keys):
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    for dotted in dotted_keys:
        section, _, key = dotted.partition(".")
        if key:
            config[section] = {k: v for k, v in config[section].items() if k != key}
        else:
            config.pop(section, None)
    return config


def test_configs_exist():
    assert MAIN in ALL_CONFIGS and len(GRPO_CONFIGS) >= 2


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_config_loads_and_rewards_exist(path):
    config = load_config(path)
    if "rewards" not in config:
        return
    weights, settings = reward_spec(config)
    assert set(weights) <= set(build_reward_functions())
    assert isinstance(settings, RewardSettings)
    assert all(weight >= 0 for weight in weights.values())


@pytest.mark.parametrize("path", GRPO_CONFIGS, ids=lambda p: p.name)
def test_run_name_matches_file_name(path):
    assert load_config(path)["run_name"] == path.stem


@pytest.mark.parametrize("path", GRPO_CONFIGS, ids=lambda p: p.name)
def test_batch_is_a_multiple_of_the_group_size(path):
    grpo = load_config(path)["grpo"]
    generation_batch = grpo["per_device_train_batch_size"] * grpo["gradient_accumulation_steps"]
    assert generation_batch % grpo["num_generations"] == 0


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("ablation_no_answer_correctness", {"rewards.weights"}),
        ("ablation_no_quote_grounding", {"rewards.weights"}),
        ("ablation_no_chunk_routing", {"rewards.weights"}),
        ("ablation_no_answer_faithfulness", {"rewards.weights"}),
        ("ablation_kl_beta", {"grpo.beta"}),
        ("ablation_dr_grpo", {"grpo.loss_type", "grpo.scale_rewards"}),
    ],
)
def test_ablations_differ_from_the_main_config_only_where_stated(name, changed):
    main = load_config(MAIN)
    ablation = load_config(CONFIG_DIR / f"{name}.yaml")
    ignored = ("run_name", "config_path", *changed)
    assert _without(ablation, *ignored) == _without(main, *ignored)
    for dotted in changed:
        section, _, key = dotted.partition(".")
        assert ablation[section][key] != main[section][key]


def test_weight_ablations_zero_exactly_one_reward():
    main_weights, _ = reward_spec(load_config(MAIN))
    for path in CONFIG_DIR.glob("ablation_no_*.yaml"):
        weights, _ = reward_spec(load_config(path))
        removed = path.stem.removeprefix("ablation_no_")
        assert weights[removed] == 0.0
        assert {k: v for k, v in weights.items() if k != removed} == {
            k: v for k, v in main_weights.items() if k != removed
        }


def test_training_and_evaluation_use_the_same_reward_definition():
    train_weights, train_settings = reward_spec(load_config(MAIN))
    eval_weights, eval_settings = reward_spec(load_config(CONFIG_DIR / "eval.yaml"))
    assert train_settings == eval_settings
    assert {k: v for k, v in train_weights.items() if v > 0} == eval_weights


def test_overrides():
    config = apply_overrides({"grpo": {"max_steps": 500}}, ["grpo.max_steps=3", "model.dtype=float32", "a.b=null"])
    assert config == {"grpo": {"max_steps": 3}, "model": {"dtype": "float32"}, "a": {"b": None}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])
