import json


def test_cli_json_is_ascii_safe_for_legacy_windows_consoles():
    rendered = json.dumps(
        {"path": r"C:\Users\Darrenxo\OneDrive\桌面\RA"},
        ensure_ascii=True,
        indent=2,
    )

    rendered.encode("cp1252")
    assert "\\u684c\\u9762" in rendered
