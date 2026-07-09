from legal_pilot.environment import _python_supported


def test_python_support_matches_project_metadata():
    assert _python_supported((3, 11))
    assert _python_supported((3, 12))
    assert not _python_supported((3, 10))
    assert not _python_supported((3, 13))
