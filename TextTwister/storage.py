"""Handwritten ASCII save format and paired resource implementations.

Spec: TextTwister/IMPLEMENTATION_SPEC.md#resource-policy.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .data.rules import MAX_NAME_LENGTH, STAGE_TIME
from .data.save_format import HEADER, MAX_BYTES, MAX_DIGITS, MAX_RECORDS


ResourceMode = Literal["protected", "unprotected"]


def valid_name(name: str) -> bool:
    """Validate a trimmed ASCII player name before serializing any record."""

    return (
        1 <= len(name) <= MAX_NAME_LENGTH
        and name == name.strip()
        and all("a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9" or c == " " for c in name)
    )


def unsigned(text: str) -> int | None:
    """Convert explicitly bounded ASCII digits; malformed fields return None."""

    if not text or len(text) > MAX_DIGITS or not all("0" <= c <= "9" for c in text):
        return None

    return int(text)


@dataclass(frozen=True)
class Record:
    player: str

    words: int

    bonus: int

    stage: int

    carryover_ns: int

    finished_ns: int

    @property
    def numerator(self) -> int:

        return (self.words + self.bonus) * self.stage * (STAGE_TIME + self.carryover_ns)

    @property
    def score_text(self) -> str:

        cents = (self.numerator * 100 + STAGE_TIME // 2) // STAGE_TIME

        return f"{cents // 100}.{cents % 100:02d}"


def ranked(records: list[Record]) -> list[Record]:
    """Keep the best bounded set, comparing exact score numerators."""

    return sorted(records, key=lambda r: (-r.numerator, r.finished_ns))[:MAX_RECORDS]


def parse_records(data: bytes) -> tuple[list[Record], str]:
    """Parse the handwritten versioned format, returning records or a line error."""

    # Spec: TextTwister/IMPLEMENTATION_SPEC.md#leaderboard-format.
    # Validate the envelope before decoding or converting numeric fields.
    if len(data) > MAX_BYTES:
        return [], "Leaderboard exceeds 64 KiB"

    if any(byte > 127 for byte in data):
        return [], "Leaderboard must be ASCII"

    lines = data.decode("ascii").splitlines()

    if not lines or lines[0] != HEADER:
        return [], "Unknown leaderboard header/version"

    records: list[Record] = []

    # Each record validates its complete shape before numeric unpacking.
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("|")

        error = f"Leaderboard line {number}: invalid record"

        if len(fields) != 6 or not valid_name(fields[0]):
            return [], error

        values = [unsigned(field) for field in fields[1:]]

        if any(value is None for value in values):
            return [], error

        words, bonus, stage, carry, finished = (value for value in values if value is not None)

        if stage < 1 or bonus > words or len(records) >= MAX_RECORDS:
            return [], error

        records.append(Record(fields[0], words, bonus, stage, carry, finished))

    return ranked(records), ""


def serialize_records(records: list[Record]) -> bytes:
    """Emit validated ASCII records without a serialization library."""

    lines = [HEADER]

    for r in ranked(records):
        lines.append(f"{r.player}|{r.words}|{r.bonus}|{r.stage}|{r.carryover_ns}|{r.finished_ns}")

    data = ("\n".join(lines) + "\n").encode("ascii")

    _, error = parse_records(data)

    if error:
        raise ValueError(error)

    return data


def read_unprotected(path: Path, *, limit: int | None = None) -> tuple[bytes | None, str]:
    """No authored try blocks. IO failures propagate to the caller."""

    with path.open("rb") as source:
        return source.read(-1 if limit is None else limit), ""


def read_protected(path: Path, *, limit: int | None = None) -> tuple[bytes | None, str]:

    try:
        return read_unprotected(path, limit=limit)

    except OSError as error:
        return None, str(error)


def write_unprotected(path: Path, data: bytes) -> None:
    """Atomic success path; an error may leave the temporary file behind."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".leaderboard-", suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)

        output.write(data)

    temporary.replace(path)

    # Protected alternative pattern:
    # try:
    #     output.write(data)
    # finally:
    #     output.close()


def remove_protected(path: Path) -> str:

    try:
        path.unlink(missing_ok=True)

        return ""

    except OSError as error:
        return str(error)


def write_protected(path: Path, data: bytes) -> str:

    temporary: Path | None = None

    error_text = ""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".leaderboard-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)

            output.write(data)

        temporary.replace(path)

    except OSError as error:
        error_text = str(error)

    finally:
        if temporary is not None:
            cleanup_error = remove_protected(temporary)

            error_text = error_text or cleanup_error

    return error_text


class Leaderboard:
    def __init__(self, path: Path, mode: ResourceMode = "protected") -> None:

        if mode not in ("protected", "unprotected"):
            raise ValueError("unknown resource mode")

        self.path, self.mode = path, mode

        self.records: list[Record] = []

        self.error = ""

        if mode == "protected":
            self._load_protected()

        else:
            self._load_unprotected()

    def _load_unprotected(self) -> None:

        if self.path.exists():
            data, self.error = read_unprotected(self.path, limit=MAX_BYTES + 1)

            if data is not None:
                self.records, self.error = parse_records(data)

    def _load_protected(self) -> None:

        try:
            self._load_unprotected()

        except OSError as error:
            self.error = str(error)

    def add(self, record: Record) -> bool:

        if self.error:
            return False

        records = ranked([*self.records, record])

        data = serialize_records(records)

        if self.mode == "protected":
            self.error = write_protected(self.path, data)

        else:
            write_unprotected(self.path, data)

        if self.error:
            return False

        self.records = records

        return True
