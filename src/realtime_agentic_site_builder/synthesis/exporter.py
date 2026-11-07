"""Packages a synthesized project into a reproducible archive, entirely in memory.

Archives come out bit for bit the same for equal bundles: entries are sorted, and timestamps,
ownership, permissions and the creating platform are fixed. Compressed bytes additionally depend on
the zlib the interpreter was built with, so reproducing a hash across machines takes the same zlib.
"""

import gzip
import hashlib
import io
import tarfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.synthesis.engine import synthesize_project

EPOCH_DATE_TIME: Final = (1980, 1, 1, 0, 0, 0)  # The earliest moment a ZIP entry can record.
EPOCH_SECONDS: Final = 315_532_800  # The same moment, in UTC.
FILE_MODE: Final = 0o644
CHUNK_BYTES: Final = 64 * 1024
_COMPRESSION_LEVEL: Final = 9
_ZIP_UNIX_SYSTEM: Final = 3


class ArchiveFormat(StrEnum):
    ZIP = "zip"
    TAR_GZ = "tar.gz"


_MEDIA_TYPES: Final[Mapping[ArchiveFormat, str]] = {
    ArchiveFormat.ZIP: "application/zip",
    ArchiveFormat.TAR_GZ: "application/gzip",
}


@dataclass(frozen=True, slots=True)
class ProjectArchive:
    filename: str
    media_type: str
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def chunks(self) -> Iterator[bytes]:
        for start in range(0, len(self.content), CHUNK_BYTES):
            yield self.content[start : start + CHUNK_BYTES]


def build_zip(entries: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(entries):
            info = zipfile.ZipInfo(path, date_time=EPOCH_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = _ZIP_UNIX_SYSTEM
            info.external_attr = FILE_MODE << 16
            archive.writestr(info, entries[path], compresslevel=_COMPRESSION_LEVEL)
    return buffer.getvalue()


def build_tar_gz(entries: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    # An empty name and a zero mtime keep the gzip header from recording either.
    with (
        gzip.GzipFile(
            filename="", mode="wb", compresslevel=_COMPRESSION_LEVEL, fileobj=buffer, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for path in sorted(entries):
            info = tarfile.TarInfo(path)
            info.size = len(entries[path])
            info.mtime = EPOCH_SECONDS
            info.mode = FILE_MODE
            archive.addfile(info, io.BytesIO(entries[path]))
    return buffer.getvalue()


def export_bundle(bundle: FullstackProjectBundle, archive_format: ArchiveFormat) -> ProjectArchive:
    """Synthesize and package a bundle, under one top-level directory named after the project."""
    project = synthesize_project(bundle)
    entries = {f"{project.slug}/{path}": text.encode() for path, text in project.files.items()}
    content = build_zip(entries) if archive_format is ArchiveFormat.ZIP else build_tar_gz(entries)
    return ProjectArchive(
        filename=f"{project.slug}.{archive_format.value}",
        media_type=_MEDIA_TYPES[archive_format],
        content=content,
    )
