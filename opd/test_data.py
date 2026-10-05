from opd.data import (
    assistant_tokens, assistant_tokens_by_begin, predictor_rows, units,
    validate)


def row(paras):
    tokens = list(range(len(paras)))
    for start, para in enumerate(paras):
        para.setdefault("start", start)
        para.setdefault("folded", False)
        para.setdefault("swap", None)
    return {"tokens": tokens, "paras": paras}


def test_search_multiturn():


    sample = row([
        {"kind": "prompt", "pid": 0},
        {"kind": "prompt", "pid": 1},
        {"kind": "assistant", "pid": 2},
        {"kind": "obs", "pid": 3, "folded": True},
        {"kind": "probe", "pid": None, "swap": [2, 3]},
        {"kind": "assistant", "pid": 4, "swap": [3]},
        {"kind": "obs", "pid": 5},
        {"kind": "probe", "pid": None, "swap": [3]},
        {"kind": "assistant", "pid": 6, "swap": [2]},
    ])
    table = units(sample)
    assert [unit.begin for unit in table] == [0, 1, 2, 3, 5, 6, 8]
    assert [unit.ctx for unit in table] == [
        [], [0], [0, 1], [0, 1, 2],
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 2, 4, 5],
    ]

    teacher = units(sample, fold=False)
    assert [unit.ctx for unit in teacher] == [list(range(i)) for i in range(7)]


def test_locomo_single_answer():
    sample = row([
        {"kind": "prompt", "pid": 0},
        {"kind": "prompt", "pid": 1},
        {"kind": "session", "pid": 2, "folded": True},
        {"kind": "session", "pid": 3},
        {"kind": "qa", "pid": 4},
        {"kind": "probe", "pid": None, "swap": [2, 3]},
        {"kind": "assistant", "pid": 5, "swap": [3]},
    ])
    table = units(sample)
    assert table[-1].ctx == [0, 1, 3, 4]


def test_duplicate_toggle_cancels():
    sample = row([
        {"kind": "prompt", "pid": 0},
        {"kind": "assistant", "pid": 1, "swap": [0, 0]},
    ])
    assert units(sample)[-1].ctx == [0]


def test_all_assistant_targets_and_trailing_observation():


    sample = {
        "tokens": list(range(14)),
        "loss_mask": [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "rollout_logprobs": [0.0] * 14,
        "paras": [
            {"start": 0, "kind": "prompt", "pid": 0, "folded": False},
            {"start": 2, "kind": "assistant", "pid": 1, "folded": False},
            {"start": 6, "kind": "obs", "pid": 2, "folded": False},
            {"start": 8, "kind": "probe", "pid": None, "folded": False},
            {"start": 9, "kind": "assistant", "pid": 3, "folded": False},
            {"start": 12, "kind": "obs", "pid": 4, "folded": False},
        ],
    }
    validate(sample)
    assert assistant_tokens_by_begin(sample) == {2: [4, 5], 9: [11]}
    assert assistant_tokens(sample) == [4, 5, 11]
    assert predictor_rows([4, 5], 2) == [1, 2]
    assert [unit.begin for unit in units(sample)] == [0, 2, 6, 9, 12]


def main():
    test_search_multiturn()
    test_locomo_single_answer()
    test_duplicate_toggle_cancels()
    test_all_assistant_targets_and_trailing_observation()
    print("opd.data dynamic swap + all-assistant targets: OK")


if __name__ == "__main__":
    main()
