from legal_pilot.io_utils import read_jsonl


def test_read_jsonl_accepts_utf8_bom(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"answer": "support"}\n')

    assert read_jsonl(path) == [{"answer": "support"}]
