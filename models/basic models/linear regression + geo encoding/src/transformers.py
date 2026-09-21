from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PostcodeRegionTransformer(BaseEstimator, TransformerMixin):
    """Derive postal sector, district, and region from a six-digit postcode."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        settings = cfg.get("feature_engineering", {}).get("postcode_transform", {})
        self.enabled = bool(settings.get("enabled", True))
        self.postcode_col = str(settings.get("postcode_col", "POSTAL_CODE"))
        self.sector_col = str(settings.get("sector_col", "POSTAL_SECTOR"))
        self.district_col = str(settings.get("district_col", "POSTAL_DISTRICT"))
        self.region_col = str(settings.get("region_col", "POSTAL_REGION"))
        self.invalid_sectors = {str(x).zfill(2) for x in settings.get("invalid_sector_prefixes", ["00"])}
        self.sector_to_district = {
            str(sector).zfill(2): str(district).zfill(2)
            for district, info in settings.get("postal_districts", {}).items()
            for sector in info.get("postal_sector", [])
        }
        self.district_to_region = {
            str(district).zfill(2): str(region)
            for region, info in settings.get("regions", {}).items()
            for district in info.get("districts", [])
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PostcodeRegionTransformer":
        return self

    @staticmethod
    def _valid_postcode(value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
        return digits if len(digits) == 6 else None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()
        if self.postcode_col not in X.columns:
            raise KeyError(f"Postal-code column '{self.postcode_col}' not found")
        output = X.copy()
        postal = output[self.postcode_col].map(self._valid_postcode)
        sector = postal.map(lambda value: value[:2] if isinstance(value, str) else None)
        sector = sector.map(lambda value: None if value in self.invalid_sectors else value)
        output[self.sector_col] = sector
        output[self.district_col] = sector.map(self.sector_to_district)
        output[self.region_col] = output[self.district_col].map(self.district_to_region)
        return output


class FloorTransformer(BaseEstimator, TransformerMixin):
    """Normalize MAX_FLOOR and create capped, band, and indicator features."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        settings = cfg.get("feature_engineering", {}).get("floor", {})
        self.enabled = bool(settings.get("enabled", True))
        self.source_col = str(settings.get("source_col", settings.get("sty_col", "MAX_FLOOR")))
        self.floor_num_col = str(settings.get("floor_num_col", "FLOOR_NUM"))
        self.floor_band_col = str(settings.get("floor_band_col", "FLOOR_BAND"))
        self.is_high_floor_col = str(settings.get("is_high_floor_col", "IS_HIGH_FLOOR"))
        self.is_basement_col = str(settings.get("is_basement_col", "IS_BASEMENT"))
        self.basement_strategy = str(settings.get("basement_strategy", "to_zero")).lower()
        self.missing_impute_value = float(settings.get("missing_impute_value", 1))
        self.high_floor_threshold = float(settings.get("high_floor_threshold", 11))
        self.cap_enabled = bool(settings.get("cap_enabled", True))
        self.cap_value = float(settings.get("cap_value", 40))
        self.bands = settings.get("bands", [])
        self._basement_pattern = re.compile(r"^B(\d+)$", re.IGNORECASE)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FloorTransformer":
        return self

    def _normalize(self, value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            value = value.strip()
            match = self._basement_pattern.match(value)
            if match:
                return -float(match.group(1))
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = X.copy()
        if self.source_col not in output.columns:
            raise KeyError(f"Floor column '{self.source_col}' not found")
        floors = output[self.source_col].map(self._normalize).fillna(self.missing_impute_value)
        is_basement = (floors < 0).astype("int8")
        if self.basement_strategy in {"to_zero", "flag"}:
            floors = floors.mask(floors < 0, 0)
        elif self.basement_strategy == "to_ground":
            floors = floors.mask(floors < 0, self.missing_impute_value)
        elif self.basement_strategy != "keep":
            raise ValueError(f"Unknown basement_strategy: {self.basement_strategy}")
        output[self.floor_num_col] = floors.astype(float)
        if not self.enabled:
            return output
        if self.cap_enabled:
            output[f"{self.floor_num_col}_CAPPED"] = floors.clip(upper=self.cap_value)
        output[self.is_high_floor_col] = (floors >= self.high_floor_threshold).astype("int8")
        if self.basement_strategy == "flag":
            output[self.is_basement_col] = is_basement
        if self.bands:
            def band(value: float) -> Optional[str]:
                for item in self.bands:
                    upper = item.get("max")
                    if value >= float(item["min"]) and (upper is None or value <= float(upper)):
                        return str(item["name"])
                return None
            output[self.floor_band_col] = floors.map(band).astype("string")
        return output
