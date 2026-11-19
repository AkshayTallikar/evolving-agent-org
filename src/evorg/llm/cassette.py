"""Record LLM calls to a JSONL cassette and replay them later, so a paid run can be repeated free.

``RecordingBackend(inner, path)`` forwards each call to ``inner`` and appends the request and
the response to the cassette. ``ReplayBackend(path)`` answers calls from a cassette and never
calls a model. A miss raises ``CassetteMiss``; nothing is guessed.

Cassette format
---------------
One JSON object per line, written with sorted keys:

    {"hash": "<hex>", "stream": "<label>", "request": {...}, "response": {...}}

``request`` holds the hashed fields plus ``"v"``, the hash version. ``response`` holds
``text``, ``input_tokens``, ``output_tokens`` and ``cost_usd`` exactly as the inner backend
returned them. Only successful calls are recorded; a call that raised leaves no line.

Request hash
------------
``request_hash`` is the sha256 hex digest of the canonical JSON (sorted keys, no spaces, UTF-8)
of ``request_payload``: ``v``, ``role_kind``, ``model``, ``system``, ``prompt``, ``task_id``,
``attempt`` and ``metadata``. ``metadata`` is included, so the architect's genome and count
are part of the key. Metadata must be JSON-serialisable; otherwise ``TypeError`` is raised
before the inner backend is called, so a paid call is never made and then lost. The digest
does not use Python's ``hash()``, which varies between processes.

Ordering and streams
--------------------
Two requests with the same hash may get different answers: a real model is not deterministic,
and two simulator seeds answer the same request differently. So each line also carries a
``stream`` label, chosen by the caller (the CLI uses one label per simulator seed). Lookups use
``(stream, hash)``. Within one key, answers are kept in the order they were recorded, and
replay returns them in that order: the n-th call with a given key gets the n-th recorded
answer. After the last recorded answer, a further call is a miss, not a repeat.

Concurrency
-----------
Writers that share a path share one lock, so lines from several threads, or from several
``RecordingBackend`` objects, never interleave. With ``workers > 1`` the interleaving of calls
that share a key follows thread timing, so the order of duplicate answers can differ between
runs. The answers themselves are still the recorded ones.

Cost
----
A replayed response carries the cost recorded in the cassette, so reports show what the paid
run cost. ``ReplayBackend.name`` is ``"replay:<path>"``, which appears in ``run.json`` and in
the report, and the CLI says that no API call was made.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evorg.llm.base import LLMBackend, LLMRequest, LLMResponse

HASH_VERSION = 1


class CassetteError(RuntimeError):
    """A cassette cannot be read, or one of its lines does not match its own hash."""


class CassetteMiss(CassetteError):
    """Replay was asked for a request the cassette has no unused answer for."""

    def __init__(
        self,
        path: str,
        stream: str,
        request: LLMRequest,
        digest: str,
        recorded: int,
    ) -> None:
        self.path = path
        self.stream = stream
        self.role_kind = request.role_kind
        self.task_id = request.task_id
        self.digest = digest
        self.recorded = recorded
        where = f"role {request.role_kind!r}"
        if request.task_id is not None:
            where += f" on task {request.task_id!r}"
        if recorded:
            reason = f"all {recorded} recorded answers were already replayed"
        else:
            reason = "no answer was recorded for it"
        stream_part = f" (stream {stream!r})" if stream else ""
        super().__init__(
            f"cassette miss: {where}, hash {digest}, in {path}{stream_part}: {reason}. "
            "The run differs from the recorded one, or the cassette is from another "
            "configuration."
        )


def request_payload(request: LLMRequest) -> dict[str, Any]:
    """The fields that determine the answer, as a plain dict. See the module docstring."""
    return {
        "v": HASH_VERSION,
        "role_kind": request.role_kind,
        "model": request.model,
        "system": request.system,
        "prompt": request.prompt,
        "task_id": request.task_id,
        "attempt": request.attempt,
        "metadata": request.metadata,
    }


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_hash(request: LLMRequest) -> str:
    """Stable sha256 hex digest of ``request``. Raises TypeError for unserialisable metadata."""
    return _digest(request_payload(request))


_path_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.abspath(path)
    with _registry_lock:
        return _path_locks.setdefault(key, threading.Lock())


class RecordingBackend:
    """Forwards each call to ``inner`` and appends the request and response to ``path``.

    The name is ``inner.name``, so a recorded run reports the same backend as an unrecorded
    one. The cassette file is created on the first write; an existing file is appended to.
    """

    def __init__(self, inner: LLMBackend, path: str | os.PathLike[str], stream: str = "") -> None:
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self.stream = stream
        self._lock = _lock_for(self.path)

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = request_payload(request)  # fails before any paid call if unserialisable
        digest = _digest(payload)
        response = self.inner.complete(request)
        line = json.dumps(
            {
                "hash": digest,
                "stream": self.stream,
                "request": payload,
                "response": {
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": float(response.cost_usd),
                },
            },
            sort_keys=True,
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return response


class Cassette:
    """A cassette file read into memory: recorded answers grouped by ``(stream, hash)``."""

    def __init__(
        self, path: str | os.PathLike[str], entries: dict[tuple[str, str], list[LLMResponse]]
    ) -> None:
        self.path = str(path)
        self.entries = entries

    @property
    def size(self) -> int:
        """Number of recorded calls in the file, over all streams."""
        return sum(len(answers) for answers in self.entries.values())

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Cassette:
        """Read and check every line. Raises CassetteError for a bad or tampered line."""
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise CassetteError(f"cannot read cassette {source}: {exc}") from exc
        entries: dict[tuple[str, str], list[LLMResponse]] = {}
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                payload = entry["request"]
                digest = entry["hash"]
                stream = entry["stream"]
                recorded = entry["response"]
                response = LLMResponse(
                    text=str(recorded["text"]),
                    input_tokens=int(recorded["input_tokens"]),
                    output_tokens=int(recorded["output_tokens"]),
                    cost_usd=float(recorded["cost_usd"]),
                )
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise CassetteError(
                    f"{source}: line {number} is not a cassette entry: {exc}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("v") != HASH_VERSION:
                raise CassetteError(f"{source}: line {number} uses another hash version")
            if _digest(payload) != digest:
                raise CassetteError(f"{source}: line {number} does not match its hash")
            entries.setdefault((str(stream), digest), []).append(response)
        return cls(source, entries)


class ReplayBackend:
    """Answers calls from a cassette, in recorded order per ``(stream, hash)``, with no model.

    ``source`` is a path or an already loaded ``Cassette``, so several streams can share one
    load. Each instance keeps its own position in the recorded answers, under a lock, so it
    can be used from several threads. The name is ``"replay:<path>"``.
    """

    def __init__(self, source: Cassette | str | os.PathLike[str], stream: str = "") -> None:
        cassette = source if isinstance(source, Cassette) else Cassette.load(source)
        self.cassette = cassette
        self.name = f"replay:{cassette.path}"
        self.stream = stream
        self._used: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        digest = request_hash(request)
        recorded = self.cassette.entries.get((self.stream, digest), [])
        with self._lock:
            used = self._used.get(digest, 0)
            if used >= len(recorded):
                raise CassetteMiss(self.cassette.path, self.stream, request, digest, len(recorded))
            self._used[digest] = used + 1
        return recorded[used]
