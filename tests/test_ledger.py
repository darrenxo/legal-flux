import json

from legal_pilot.ledger import JsonlLedger, make_run_hash


def test_ledger_appends_and_resumes(tmp_path):
    path = tmp_path / "records.jsonl"
    ledger = JsonlLedger(path)
    run_hash = make_run_hash(
        dataset="legalhk",
        case_id="c1",
        variant_id="original",
        condition="direct",
        prompt_hash="abc",
        model_digest="sha256:x",
        seed=7,
        sample_index=0,
    )
    assert not ledger.contains(run_hash)
    ledger.append({"run_hash": run_hash, "status": "ok"})
    assert ledger.contains(run_hash)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"run_hash": run_hash, "status": "ok"}]


def test_ledger_retries_errors_but_skips_successes(tmp_path):
    path = tmp_path / "records.jsonl"
    run_hash = "same-job"
    ledger = JsonlLedger(path)
    ledger.append({"run_hash": run_hash, "status": "error"})
    assert not ledger.contains(run_hash)

    ledger.append({"run_hash": run_hash, "status": "ok"})
    assert ledger.contains(run_hash)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_ledger_discards_only_truncated_final_record(tmp_path):
    path = tmp_path / "generations.jsonl"
    complete = {"run_hash": "complete", "status": "ok"}
    path.write_text(
        json.dumps(complete) + '\n{"run_hash":"truncated',
        encoding="utf-8",
    )

    ledger = JsonlLedger(path)
    assert ledger.contains("complete")

    recovered = {"run_hash": "recovered", "status": "ok"}
    ledger.append(recovered)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [complete, recovered]


def test_ledger_preserves_valid_unterminated_final_record(tmp_path):
    path = tmp_path / "generations.jsonl"
    complete = {"run_hash": "complete", "status": "ok"}
    path.write_text(json.dumps(complete), encoding="utf-8")

    ledger = JsonlLedger(path)

    assert ledger.contains("complete")
    assert path.read_bytes().endswith(b"\n")
    assert json.loads(path.read_text(encoding="utf-8")) == complete
