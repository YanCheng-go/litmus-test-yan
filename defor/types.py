
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

FILENAME_TIME_RE = re.compile(
    r"(?:^|[/\\])[^/\\]*_(?P<year>\d{4})_(?P<half>[12])[^/\\]*\.(?:tif|tiff)$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class Bands:
    """1-based band indices (rasterio convention)."""
    nir: int = 5
    swir1: int = 6
    blue: int = 2
    green: int = 3
    red: int = 4

@dataclass(frozen=True)
class Options:
    index: str = "swir_nir"          # "swir_nir" or "nir_minus_red"
    percentile_thr: float = 50.0     # fallback percentile when Otsu fails
    persistence: int = 0             # consecutive non-forest steps
    require_nonforest_until_end: bool = False
    use_water_mask: bool = False
    use_cloud_mask: bool = False

@dataclass(frozen=True)
class TimeStamp:
    year: int
    half: int  # 1 or 2

    @property
    def encoded(self) -> int:
        return self.year * 10 + self.half

    @staticmethod
    def from_path(p: Path) -> "TimeStamp | None":
        m = FILENAME_TIME_RE.search(str(p))
        if not m:
            return None
        return TimeStamp(year=int(m.group("year")), half=int(m.group("half")))
