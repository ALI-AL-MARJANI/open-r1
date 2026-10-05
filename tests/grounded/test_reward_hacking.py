"""Degenerate policies must not score well.

Each policy below ignores the question. The first version of this reward suite gave
"always abstain" the maximum reward on every example, because abstention was rewarded
without looking at the answerability label and no reward compared the answer with the
gold answer.

`reward_fraction` is the total weighted reward over a balanced set of answerable and
unanswerable questions, divided by the maximum attainable total. Note that any
policy that always abstains is right on the unanswerable half, which bounds how low
its score can go on a balanced set; the per-subset assertions make the intent explicit.
"""

import json

import pytest

from open_r1.grounded.legacy_v0 import legacy_reward
from open_r1.grounded.policies import ALWAYS_ABSTAIN, ALWAYS_ANSWER, TRIVIAL_POLICIES
from open_r1.grounded.rewards import weighted_reward

from .conftest import columns_of

TRIVIAL_POLICY_CEILING = 0.5


POLICIES = list(TRIVIAL_POLICIES.values())


def reward_fraction(policy, records, weights):
    totals = weighted_reward([policy(r) for r in records], columns_of(records), weights)
    return sum(t for t, _ in totals) / sum(m for _, m in totals)


def test_the_set_is_balanced(records):
    assert sum(r["is_answerable"] for r in records) * 2 == len(records)


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.__name__)
def test_trivial_policy_stays_below_the_ceiling(policy, records, reward_weights):
    assert reward_fraction(policy, records, reward_weights) < TRIVIAL_POLICY_CEILING


@pytest.mark.parametrize("policy", [TRIVIAL_POLICIES[n] for n in ALWAYS_ABSTAIN], ids=lambda p: p.__name__)
def test_abstaining_earns_only_the_format_reward_on_answerable_questions(policy, records, reward_weights):
    answerable = [r for r in records if r["is_answerable"]]
    fraction = reward_fraction(policy, answerable, reward_weights)
    assert fraction <= reward_weights["format"] / sum(reward_weights.values()) + 1e-9


@pytest.mark.parametrize("policy", [TRIVIAL_POLICIES[n] for n in ALWAYS_ANSWER], ids=lambda p: p.__name__)
def test_always_answering_earns_only_the_format_reward_on_unanswerable_questions(policy, records, reward_weights):
    unanswerable = [r for r in records if not r["is_answerable"]]
    maximum = reward_weights["format"] + reward_weights["answer_correctness"]
    assert reward_fraction(policy, unanswerable, reward_weights) <= reward_weights["format"] / maximum + 1e-9


def test_reference_policy_reaches_the_maximum(records, reward_weights):
    assert reward_fraction(lambda r: r["target"], records, reward_weights) == pytest.approx(1.0)


def test_every_trivial_policy_is_far_below_the_reference(records, reward_weights):
    best_trivial = max(reward_fraction(p, records, reward_weights) for p in POLICIES)
    assert best_trivial < 0.5 * reward_fraction(lambda r: r["target"], records, reward_weights)


def test_first_reward_version_was_maximised_by_always_abstaining(records):
    """Regression documentation: the shortcut that motivated the redesign."""
    scores = []
    for record in records:
        context = " ".join(json.loads(record["context_chunks"]).values())
        scores.append(legacy_reward(TRIVIAL_POLICIES["always_abstain"](record), context))
    assert sum(t for t, _ in scores) == pytest.approx(sum(m for _, m in scores))
