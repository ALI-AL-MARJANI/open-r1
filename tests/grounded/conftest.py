"""Shared fixtures: a small hand-written SQuAD-style corpus, no network access."""

import pytest

from open_r1.grounded.data import build_title_index, convert_squad_example

REWARD_WEIGHTS = {
    "format": 0.5,
    "answer_correctness": 2.0,
    "quote_grounding": 1.5,
    "chunk_routing": 1.0,
    "answer_faithfulness": 0.5,
}

_NORMANDY_1 = (
    "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region "
    "in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway. "
    "Their leader Rollo agreed to swear fealty to King Charles III of West Francia."
)
_NORMANDY_2 = (
    "The Norman dynasty had a major political, cultural and military impact on medieval Europe. "
    "Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy. "
    "The duchy later passed to the Plantagenet kings of England in the twelfth century."
)
_NORMANDY_3 = (
    "Norman architecture typically stands out as a new stage in the architectural history of the regions "
    "they subdued. They spread a unique Romanesque idiom to England and Italy. "
    "Many castles were built with massive proportions and rounded arches."
)
_OCEAN_1 = (
    "The Amazon rainforest covers most of the Amazon basin of South America. This basin encompasses "
    "7,000,000 square kilometres, of which 5,500,000 square kilometres are covered by the rainforest. "
    "The majority of the forest is contained within Brazil, with 60% of the rainforest."
)
_OCEAN_2 = (
    "Wet tropical forests are the most species-rich biome on the planet. One in ten known species in the "
    "world lives in the Amazon rainforest. The region is home to about 2.5 million insect species."
)


_OCEAN_3 = (
    "Deforestation is the conversion of forested areas to non-forested areas. Cattle ranching is a leading "
    "cause of land clearing in the region. Satellite monitoring began in the late 1980s."
)


def _raw(example_id, title, context, question, answers):
    return {
        "id": example_id,
        "title": title,
        "context": context,
        "question": question,
        "answers": {"text": answers, "answer_start": [context.find(a) for a in answers]},
    }


RAW_EXAMPLES = [
    _raw("q1", "Normans", _NORMANDY_1, "In what country is Normandy located?", ["France"]),
    _raw("q2", "Normans", _NORMANDY_1, "Where did the Norse originate?", ["Denmark, Iceland and Norway"]),
    _raw("q3", "Normans", _NORMANDY_2, "Who ruled the Kingdom of Sicily?", ["Roger II"]),
    _raw("q4", "Normans", _NORMANDY_3, "What did the Normans spread?", ["a unique Romanesque idiom", "Romanesque"]),
    _raw("q5", "Amazon", _OCEAN_1, "Which country contains most of the forest?", ["Brazil"]),
    _raw("q6", "Amazon", _OCEAN_2, "How many insect species live there?", ["about 2.5 million", "2.5 million"]),
    _raw("u1", "Normans", _NORMANDY_1, "Who gave their name to Brittany in the 12th century?", []),
    _raw("u2", "Normans", _NORMANDY_2, "Which dynasty ruled medieval Japan?", []),
    _raw("u3", "Normans", _NORMANDY_3, "What style was replaced by Norman castles in Spain?", []),
    _raw("u4", "Amazon", _OCEAN_1, "What share of the Congo rainforest is in Brazil?", []),
    _raw("u5", "Amazon", _OCEAN_2, "How many mammal species live on the planet?", []),
    _raw("u6", "Amazon", _OCEAN_3, "Which crop is the leading cause of land clearing in Asia?", []),
]


@pytest.fixture(scope="session")
def raw_examples():
    return RAW_EXAMPLES


@pytest.fixture(scope="session")
def records():
    """Balanced set of 6 answerable and 6 unanswerable records with 2 distractors each."""
    index = build_title_index(RAW_EXAMPLES)
    return [convert_squad_example(ex, index, n_distractors=2, seed=0) for ex in RAW_EXAMPLES]


@pytest.fixture(scope="session")
def reward_weights():
    return dict(REWARD_WEIGHTS)


def columns_of(records):
    """Dataset columns as TRL passes them to reward functions (one list per column)."""
    keys = ("context_chunks", "gold_answers", "gold_chunk_ids", "is_answerable")
    return {key: [r[key] for r in records] for key in keys}


def as_chat(texts):
    """Wrap raw strings as TRL conversational completions."""
    return [[{"role": "assistant", "content": text}] for text in texts]
