"""Portable retrieval evidence values and content digests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    source_uri: str
    final_uri: str
    observed_at: datetime
    content_sha256: str
    media_type: str
    locator: str

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("evidence observed_at must be timezone-aware")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("evidence content_sha256 must be lowercase SHA-256")
        if not all((self.source_uri, self.final_uri, self.media_type, self.locator)):
            raise ValueError("evidence receipt fields must not be empty")
