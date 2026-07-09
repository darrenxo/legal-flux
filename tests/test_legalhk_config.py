from pathlib import Path

from legal_pilot.config import load_config, resolve_path


def test_legalhk_config_is_isolated_and_has_expected_counts():
    config_path = Path(__file__).parents[1] / "configs" / "legalhk_only.yaml"
    config = load_config(config_path)

    assert config["data"]["datasets"] == ["legalhk"]
    assert config["data"]["smoke_cases"] == 5
    assert config["data"]["evaluation_cases"] == 64
    assert config["data"]["oracle_cases"] == 24
    assert config["data"]["sampling_control_cases"] == 12
    assert config["data"]["sampling_control_repeats"] == 3
    assert config["data"]["smoke_case_ids"] == [
        "legalhk-2185",
        "legalhk-5875",
        "legalhk-864",
        "legalhk-2753",
        "legalhk-15673",
    ]
    excluded = config["data"]["excluded_evaluation_case_ids"]
    assert {"legalhk-5096", "legalhk-2540"} <= set(excluded)
    assert len(excluded) == len(set(excluded))
    assert str(resolve_path(config, "processed_dir")).endswith(
        r"data\processed\legalhk_only"
    )
    assert str(resolve_path(config, "runs_dir")).endswith(
        r"runs\legalhk_only"
    )
