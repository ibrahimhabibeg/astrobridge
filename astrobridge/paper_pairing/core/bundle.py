"""
The central state container for the crossmatch pipeline.

Manages 4 DataFrames (persisted as Parquet):
  - hf_objects:       Astronomical observations from a Hugging Face MMU catalog.
  - simbad_objects:   Matched objects resolved through SIMBAD.
  - ads_papers:       Paper metadata fetched from NASA ADS.
  - relationships:    Edge table linking the three entity tables.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_HF_OBJECTS_COLUMNS = ["object_id", "ra", "dec"]
_SIMBAD_OBJECTS_COLUMNS = ["main_id", "coo_bibcode"]
_ADS_PAPERS_COLUMNS = ["bibcode", "paper_title", "abstract", "doi", "preprint_url", "keyword"]
_RELATIONSHIPS_COLUMNS = ["hf_id", "simbad_main_id", "bibcode"]

_TABLE_FILE_MAP = {
    "hf_objects": "hf_objects.parquet",
    "simbad_objects": "simbad_objects.parquet",
    "ads_papers": "ads_papers.parquet",
    "relationships": "relationships.parquet",
}

_FK_MAP = {
    "hf_id": ("hf_objects", "object_id"),
    "simbad_main_id": ("simbad_objects", "main_id"),
    "bibcode": ("ads_papers", "bibcode"),
}


class CrossmatchBundle:
    """Immutable-schema, mutable-data state manager for the crossmatch pipeline.

    Parameters
    ----------
    hf_objects : pd.DataFrame
        Must contain at least ``object_id``, ``ra``, ``dec``.
    simbad_objects : pd.DataFrame
        Must contain at least ``main_id``, ``coo_bibcode``.
    ads_papers : pd.DataFrame
        Must contain at least ``bibcode``, ``paper_title``, ``abstract``,
        ``doi``, ``preprint_url``.
    relationships : pd.DataFrame
        Must contain at least ``hf_id``, ``simbad_main_id``, ``bibcode``.
    """

    def __init__(
        self,
        hf_objects: pd.DataFrame,
        simbad_objects: pd.DataFrame,
        ads_papers: pd.DataFrame,
        relationships: pd.DataFrame,
    ) -> None:
        self.hf_objects = self._validate(hf_objects, _HF_OBJECTS_COLUMNS, "hf_objects")
        self.simbad_objects = self._validate(
            simbad_objects, _SIMBAD_OBJECTS_COLUMNS, "simbad_objects"
        )
        self.ads_papers = self._validate(ads_papers, _ADS_PAPERS_COLUMNS, "ads_papers")
        self.relationships = self._validate(
            relationships, _RELATIONSHIPS_COLUMNS, "relationships"
        )

    def copy(self) -> CrossmatchBundle:
        """Return a deep copy of this bundle."""
        return CrossmatchBundle(
            hf_objects=self.hf_objects.copy(),
            simbad_objects=self.simbad_objects.copy(),
            ads_papers=self.ads_papers.copy(),
            relationships=self.relationships.copy(),
        )

    @staticmethod
    def _validate(
        df: pd.DataFrame, required_cols: Sequence[str], table_name: str
    ) -> pd.DataFrame:
        """Ensure *required_cols* are present; extra augmentation columns are allowed."""
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Table '{table_name}' is missing required columns: {missing}"
            )
        return df.reset_index(drop=True)

    def synchronize_state(self) -> None:
        """Cascade-delete orphaned rows across all tables."""
        # Prune relationship edges pointing to missing entities
        mask = pd.Series(True, index=self.relationships.index)
        for fk_col, (table_name, pk_col) in _FK_MAP.items():
            entity_table: pd.DataFrame = getattr(self, table_name)
            valid_keys = set(entity_table[pk_col])
            mask &= self.relationships[fk_col].isin(valid_keys)

        rows_before = len(self.relationships)
        self.relationships = self.relationships.loc[mask].reset_index(drop=True)
        dropped = rows_before - len(self.relationships)
        if dropped:
            logger.info("synchronize_state pass-1: dropped %d orphaned edges", dropped)

        # Prune entity rows with no remaining edges
        for fk_col, (table_name, pk_col) in _FK_MAP.items():
            entity_table = getattr(self, table_name)
            referenced_keys = set(self.relationships[fk_col])
            keep = entity_table[pk_col].isin(referenced_keys)
            entity_dropped = (~keep).sum()
            if entity_dropped:
                logger.info(
                    "synchronize_state pass-2: dropped %d orphaned rows from '%s'",
                    entity_dropped,
                    table_name,
                )
            setattr(self, table_name, entity_table.loc[keep].reset_index(drop=True))

        # Deduplicate relationships
        before = len(self.relationships)
        self.relationships = self.relationships.drop_duplicates().reset_index(drop=True)
        deduped = before - len(self.relationships)
        if deduped:
            logger.info("synchronize_state pass-3: removed %d duplicate edges", deduped)

    def save(self, directory: str | Path) -> Path:
        """Persist all 4 tables to Parquet files in directory.

        Returns the resolved directory path.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for table_name, filename in _TABLE_FILE_MAP.items():
            table: pd.DataFrame = getattr(self, table_name)
            table.to_parquet(directory / filename, index=False, engine="pyarrow")

        logger.info("CrossmatchBundle saved to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CrossmatchBundle":
        """Load a previously-saved bundle from *directory*."""
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {directory}")

        tables = {}
        for table_name, filename in _TABLE_FILE_MAP.items():
            path = directory / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing table file '{filename}' in {directory}"
                )
            tables[table_name] = pd.read_parquet(path, engine="pyarrow")

        return cls(**tables)

    def summary(self) -> str:
        """Human-readable snapshot of the bundle."""
        lines = ["CrossmatchBundle Summary", "=" * 40]
        for table_name in _TABLE_FILE_MAP:
            table: pd.DataFrame = getattr(self, table_name)
            cols = list(table.columns)
            lines.append(f"  {table_name}: {len(table)} rows, columns={cols}")
        return "\n".join(lines)
