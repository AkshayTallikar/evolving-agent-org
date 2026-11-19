import json
import random
import threading
from pathlib import Path

import pytest

from evorg.bench import load_benchmark, task_index
from evorg.cli import EXIT_ERROR, EXIT_USAGE, main
from evorg.evolution import EvolutionConfig, evolve, run_simulated
from evorg.llm import SimulatedBackend
from evorg.llm.base import LLMRequest, LLMResponse
from evorg.llm.cassette import (
    Cassette,
    CassetteError,
    CassetteMiss,
    RecordingBackend,
    ReplayBackend,
    request_hash,
)
from evorg.mutation import Architect
from evorg.report import SIM_NOTE

TINY = dict(population=4, generations=2, seed=3, eval_seeds=2)


class FakeBackend:
    """Answers each call with a numbered text, so duplicate requests get different answers."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"answer {self.calls}", input_tokens=3, output_tokens=5, cost_usd=0.25
        )


def _req(prompt: str = "Write add.", **kw: object) -> LLMRequest:
    return LLMRequest("coder", "small", "Be brief.", prompt, task_id="t1", **kw)


def _comparable(run_dict: dict) -> dict:
    """A run.json-style dict without the backend name, which replay changes on purpose."""
    return {k: v for k, v in run_dict.items() if k != "backend"}


def _record_tiny(path: Path, config: EvolutionConfig | None = None):
    config = config or EvolutionConfig(**TINY)

    def wrap(inner, stream):
        return RecordingBackend(inner, path, stream=stream)

    return run_simulated(config, wrap=wrap)


def _replay_tiny(path: Path, config: EvolutionConfig | None = None):
    config = config or EvolutionConfig(**TINY)
    cassette = Cassette.load(path)
    return run_simulated(config, wrap=lambda _inner, stream: ReplayBackend(cassette, stream))


def test_hash_matches_a_known_digest(tmp_path):
    # sha256 of the canonical JSON written out by hand, so the hash cannot drift silently.
    expected = "1a33e6880e79d4774421c005bb14e64f55c478107d2fa48db91135cec06572e0"
    request = LLMRequest("coder", "small", "Be brief.", "Write add.", task_id="t1")
    assert request_hash(request) == expected


def test_hash_depends_on_every_answer_field_and_metadata():
    base = _req()
    digests = {
        request_hash(base),
        request_hash(LLMRequest("tester", "small", "Be brief.", "Write add.", task_id="t1")),
        request_hash(LLMRequest("coder", "large", "Be brief.", "Write add.", task_id="t1")),
        request_hash(LLMRequest("coder", "small", "Be bold.", "Write add.", task_id="t1")),
        request_hash(_req(prompt="Write sub.")),
        request_hash(_req(attempt=1)),
        request_hash(LLMRequest("coder", "small", "Be brief.", "Write add.", task_id="t2")),
        request_hash(LLMRequest("coder", "small", "Be brief.", "Write add.", task_id=None)),
        request_hash(
            LLMRequest("coder", "small", "Be brief.", "Write add.", task_id="t1", metadata={"n": 1})
        ),
    }
    assert len(digests) == 9


def test_hash_ignores_metadata_key_order_and_is_hex():
    a = LLMRequest("architect", "large", "s", "p", metadata={"n": 2, "genome": {"x": 1}})
    b = LLMRequest("architect", "large", "s", "p", metadata={"genome": {"x": 1}, "n": 2})
    digest = request_hash(a)
    assert digest == request_hash(b)
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_hash_refuses_unserialisable_metadata():
    request = LLMRequest("architect", "large", "s", "p", metadata={"bad": object()})
    with pytest.raises(TypeError):
        request_hash(request)


def test_recording_refuses_unserialisable_metadata_before_paying(tmp_path):
    fake = FakeBackend()
    path = tmp_path / "c.jsonl"
    request = LLMRequest("architect", "large", "s", "p", metadata={"bad": object()})
    with pytest.raises(TypeError):
        RecordingBackend(fake, path).complete(request)
    assert fake.calls == 0
    assert not path.exists()


def test_record_then_replay_gives_an_identical_run(tmp_path):
    path = tmp_path / "run.jsonl"
    config = EvolutionConfig(**TINY)
    plain = run_simulated(config)
    recorded = _record_tiny(path, config)
    replayed = _replay_tiny(path, config)

    assert recorded.backend == "sim"
    assert replayed.backend == f"replay:{path}"
    assert _comparable(replayed.to_dict()) == _comparable(recorded.to_dict())
    assert _comparable(recorded.to_dict()) == _comparable(plain.to_dict())


def test_replayed_responses_carry_the_recorded_cost(tmp_path):
    path = tmp_path / "c.jsonl"
    fake = FakeBackend()
    RecordingBackend(fake, path).complete(_req())
    replay = ReplayBackend(path)
    response = replay.complete(_req())
    assert response == LLMResponse(text="answer 1", input_tokens=3, output_tokens=5, cost_usd=0.25)
    assert replay.name == f"replay:{path}"
    assert fake.calls == 1


def test_recorded_lines_have_hash_request_and_response(tmp_path):
    path = tmp_path / "c.jsonl"
    RecordingBackend(FakeBackend(), path, stream="s1").complete(_req())
    (line,) = path.read_text().splitlines()
    entry = json.loads(line)
    assert entry["hash"] == request_hash(_req())
    assert entry["stream"] == "s1"
    assert entry["request"]["prompt"] == "Write add."
    assert entry["response"] == {
        "text": "answer 1",
        "input_tokens": 3,
        "output_tokens": 5,
        "cost_usd": 0.25,
    }


def test_miss_raises_and_names_hash_and_role(tmp_path):
    path = tmp_path / "c.jsonl"
    RecordingBackend(FakeBackend(), path).complete(_req())
    replay = ReplayBackend(path)
    missing = _req(prompt="Write something else.")
    with pytest.raises(CassetteMiss) as info:
        replay.complete(missing)
    exc = info.value
    assert isinstance(exc, CassetteError)
    assert exc.digest == request_hash(missing)
    assert exc.role_kind == "coder" and exc.task_id == "t1"
    assert request_hash(missing) in str(exc)
    assert "coder" in str(exc) and "t1" in str(exc)


def test_replay_with_another_seed_misses_loudly(tmp_path):
    path = tmp_path / "seed3.jsonl"
    _record_tiny(path, EvolutionConfig(**TINY))
    with pytest.raises(CassetteMiss):
        _replay_tiny(path, EvolutionConfig(**{**TINY, "seed": 4}))


def test_duplicate_requests_replay_in_recorded_order(tmp_path):
    path = tmp_path / "c.jsonl"
    recorder = RecordingBackend(FakeBackend(), path)
    for _ in range(3):
        recorder.complete(_req())
    replay = ReplayBackend(path)
    texts = [replay.complete(_req()).text for _ in range(3)]
    assert texts == ["answer 1", "answer 2", "answer 3"]
    with pytest.raises(CassetteMiss, match="already replayed"):
        replay.complete(_req())


def test_streams_keep_answers_for_one_request_apart(tmp_path):
    path = tmp_path / "c.jsonl"
    RecordingBackend(FakeBackend(), path, stream="a").complete(_req())
    RecordingBackend(FakeBackend(), path, stream="b").complete(_req())
    cassette = Cassette.load(path)
    a = ReplayBackend(cassette, stream="a")
    b = ReplayBackend(cassette, stream="b")
    assert a.complete(_req()).text == "answer 1"
    assert b.complete(_req()).text == "answer 1"
    with pytest.raises(CassetteMiss):
        a.complete(_req())
    with pytest.raises(CassetteMiss, match="no answer was recorded"):
        ReplayBackend(cassette, stream="c").complete(_req())


def test_concurrent_recording_from_threads_gives_a_valid_file(tmp_path):
    path = tmp_path / "c.jsonl"
    threads_n, per_thread = 8, 25
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            # Each thread owns its own RecordingBackend; they share the file and its lock.
            backend = RecordingBackend(FakeBackend(), path, stream=f"t{i}")
            for j in range(per_thread):
                backend.complete(_req(prompt=f"thread {i} call {j}"))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    lines = path.read_text().splitlines()
    assert len(lines) == threads_n * per_thread
    for line in lines:
        json.loads(line)  # every line is one complete JSON object
    cassette = Cassette.load(path)  # and every hash matches its request
    assert cassette.size == threads_n * per_thread


def test_load_rejects_an_edited_line(tmp_path):
    path = tmp_path / "c.jsonl"
    RecordingBackend(FakeBackend(), path).complete(_req())
    path.write_text(path.read_text().replace("Write add.", "Write sub."))
    with pytest.raises(CassetteError, match="does not match its hash"):
        Cassette.load(path)


def test_load_rejects_a_malformed_line(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("{not json\n")
    with pytest.raises(CassetteError, match="line 1"):
        Cassette.load(path)


def test_load_reports_a_missing_file(tmp_path):
    with pytest.raises(CassetteError, match="cannot read"):
        Cassette.load(tmp_path / "absent.jsonl")


def test_cli_record_then_replay_round_trip(tmp_path, capsys):
    cassette = tmp_path / "tiny.jsonl"
    flags = ["--generations", "1", "--population", "3", "--eval-seeds", "2", "--seed", "3"]
    recorded_dir = tmp_path / "rec"
    replayed_dir = tmp_path / "rep"

    assert main(["run", "--record", str(cassette), *flags, "--out", str(recorded_dir)]) == 0
    capsys.readouterr()
    assert main(["run", "--replay", str(cassette), *flags, "--out", str(replayed_dir)]) == 0
    err = capsys.readouterr().err
    assert "no API calls, no money spent" in err

    rec = json.loads((recorded_dir / "run.json").read_text())
    rep = json.loads((replayed_dir / "run.json").read_text())
    assert rec["backend"] == "sim"
    assert rep["backend"] == f"replay-sim:{cassette}"
    assert _comparable(rep) == _comparable(rec)

    def strip_backend_line(text: str) -> str:
        # Only the backend line differs: a replayed simulator run keeps the simulator caveats.
        keep = [line for line in text.splitlines() if line and not line.startswith("Backend ")]
        return "\n".join(keep)

    rec_report = (recorded_dir / "report.md").read_text()
    rep_report = (replayed_dir / "report.md").read_text()
    assert SIM_NOTE in rep_report
    assert strip_backend_line(rep_report) == strip_backend_line(rec_report)


def test_cli_replays_a_real_model_style_cassette_through_the_paid_path(tmp_path, capsys):
    # A paid run records one unlabelled stream through evolve(). The simulator stands in for
    # the paid model here, so no API is called.
    cassette = tmp_path / "paid.jsonl"
    config = EvolutionConfig(generations=1, population=3, seed=3)
    recorder = RecordingBackend(SimulatedBackend(task_index(), seed=7), cassette)
    recorded = evolve(
        config,
        recorder,
        load_benchmark("train"),
        load_benchmark("holdout"),
        Architect(recorder, random.Random(config.seed + 1)),
    )
    assert Cassette.load(cassette).streams == {""}

    out = tmp_path / "rep"
    flags = ["--generations", "1", "--population", "3", "--seed", "3", "--out", str(out)]
    assert main(["run", "--replay", str(cassette), *flags]) == 0
    assert "no API calls, no money spent" in capsys.readouterr().err
    replayed = json.loads((out / "run.json").read_text())
    assert replayed["backend"] == f"replay:{cassette}"
    assert _comparable(replayed) == _comparable(recorded.to_dict())
    assert SIM_NOTE not in (out / "report.md").read_text()


def test_cli_replay_default_output_does_not_overwrite_a_sim_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cassette = tmp_path / "c.jsonl"
    flags = ["--generations", "1", "--population", "3", "--seed", "5"]
    assert main(["run", "--record", str(cassette), *flags]) == 0
    assert main(["run", "--replay", str(cassette), *flags]) == 0
    assert (tmp_path / "runs" / "sim-seed5" / "run.json").is_file()
    assert (tmp_path / "runs" / "replay-seed5" / "run.json").is_file()


def test_cli_replay_with_workers_gives_the_same_run(tmp_path):
    cassette = tmp_path / "c.jsonl"
    flags = ["--generations", "1", "--population", "3", "--eval-seeds", "2", "--seed", "3"]
    assert main(["run", "--record", str(cassette), *flags, "--out", str(tmp_path / "a")]) == 0
    assert (
        main(
            [
                "run",
                "--replay",
                str(cassette),
                *flags,
                "--workers",
                "3",
                "--out",
                str(tmp_path / "b"),
            ]
        )
        == 0
    )
    a = json.loads((tmp_path / "a" / "run.json").read_text())
    b = json.loads((tmp_path / "b" / "run.json").read_text())
    assert _comparable(a) == _comparable(b)


def test_cli_replay_miss_exits_with_an_error(tmp_path, capsys):
    cassette = tmp_path / "c.jsonl"
    flags = ["--generations", "1", "--population", "3"]
    assert (
        main(
            ["run", "--record", str(cassette), *flags, "--seed", "3", "--out", str(tmp_path / "a")]
        )
        == 0
    )
    capsys.readouterr()
    code = main(
        ["run", "--replay", str(cassette), *flags, "--seed", "4", "--out", str(tmp_path / "b")]
    )
    assert code == EXIT_ERROR
    assert "cassette miss" in capsys.readouterr().err
    assert not (tmp_path / "b").exists()


def test_cli_record_refuses_an_existing_file(tmp_path, capsys):
    cassette = tmp_path / "c.jsonl"
    cassette.write_text("")
    code = main(["run", "--record", str(cassette), "--generations", "1", "--population", "3"])
    assert code == EXIT_USAGE
    assert "exists" in capsys.readouterr().err


def test_cli_record_and_replay_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as info:
        main(["run", "--record", str(tmp_path / "a"), "--replay", str(tmp_path / "b")])
    assert info.value.code == 2


def test_cli_replay_rejects_the_anthropic_backend(tmp_path, capsys):
    cassette = tmp_path / "c.jsonl"
    cassette.write_text("")
    code = main(["run", "--backend", "anthropic", "--replay", str(cassette)])
    assert code == EXIT_USAGE
    assert "--replay" in capsys.readouterr().err


def test_simulated_backend_is_the_recorded_backend(tmp_path):
    # The recorder wraps the simulator itself: a replay of its answers equals the simulator's.
    path = tmp_path / "c.jsonl"
    sim = SimulatedBackend(task_index(), seed=0)
    request = LLMRequest("planner", "small", "Plan.", "Add two numbers.", task_id="add")
    recorded = RecordingBackend(sim, path).complete(request)
    assert ReplayBackend(path).complete(request) == recorded
