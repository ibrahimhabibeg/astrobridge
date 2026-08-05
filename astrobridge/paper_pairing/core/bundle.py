"""
The central state container for the crossmatch pipeline.

Manages DataFrames (persisted as Parquet):
  - datasets:         Dictionary of astronomical observations from multiple HF datasets.
  - simbad_objects:   Matched objects resolved through SIMBAD (source of truth coordinates).
  - ads_papers:       Paper metadata fetched from NASA ADS.
  - relationships:    Edge table linking SIMBAD objects to ADS papers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DATASET_COLUMNS = ["object_id", "ra", "dec", "simbad_main_id"]
_SIMBAD_OBJECTS_COLUMNS = ["main_id", "ra", "dec", "coo_bibcode"]
_ADS_PAPERS_COLUMNS = ["bibcode", "paper_title", "abstract", "doi", "preprint_url", "keyword"]
_RELATIONSHIPS_COLUMNS = ["simbad_main_id", "bibcode"]


class CrossmatchBundle:
    """Immutable-schema, mutable-data state manager for the crossmatch pipeline.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Each DataFrame must contain at least ``object_id``, ``ra``, ``dec``, ``simbad_main_id``.
    simbad_objects : pd.DataFrame
        Must contain at least ``main_id``, ``ra``, ``dec``, ``coo_bibcode``.
    ads_papers : pd.DataFrame
        Must contain at least ``bibcode``, ``paper_title``, ``abstract``,
        ``doi``, ``preprint_url``.
    relationships : pd.DataFrame
        Must contain at least ``simbad_main_id``, ``bibcode``.
    """

    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        simbad_objects: pd.DataFrame,
        ads_papers: pd.DataFrame,
        relationships: pd.DataFrame,
    ) -> None:
        self.datasets = {
            name: self._validate(df, _DATASET_COLUMNS, f"dataset_{name}")
            for name, df in datasets.items()
        }
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
            datasets={name: df.copy() for name, df in self.datasets.items()},
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
        """Cascade-delete orphaned rows across all tables until the graph stabilizes."""
        
        iteration = 1
        while True:
            total_dropped = 0
            
            # Prune relationships pointing to missing entities
            valid_simbad = set(self.simbad_objects["main_id"])
            valid_papers = set(self.ads_papers["bibcode"])
            
            mask = self.relationships["simbad_main_id"].isin(valid_simbad) & \
                   self.relationships["bibcode"].isin(valid_papers)
            
            rows_before = len(self.relationships)
            self.relationships = self.relationships.loc[mask].reset_index(drop=True)
            dropped_rels = rows_before - len(self.relationships)
            total_dropped += dropped_rels
            if dropped_rels:
                logger.info("synchronize_state iter %d: dropped %d orphaned edges", iteration, dropped_rels)

            # Prune papers with no relationships
            referenced_papers = set(self.relationships["bibcode"])
            keep_papers = self.ads_papers["bibcode"].isin(referenced_papers)
            papers_dropped = (~keep_papers).sum()
            total_dropped += papers_dropped
            if papers_dropped:
                logger.info("synchronize_state iter %d: dropped %d orphaned papers", iteration, papers_dropped)
                self.ads_papers = self.ads_papers.loc[keep_papers].reset_index(drop=True)
                
            # Prune SIMBAD objects missing from either side (must have both papers AND observations)
            referenced_simbad_in_rels = set(self.relationships["simbad_main_id"])
            referenced_simbad_in_obs = set()
            for df in self.datasets.values():
                referenced_simbad_in_obs.update(df["simbad_main_id"].dropna())
                
            keep_simbad = self.simbad_objects["main_id"].isin(referenced_simbad_in_rels) & \
                          self.simbad_objects["main_id"].isin(referenced_simbad_in_obs)
                          
            simbad_dropped = (~keep_simbad).sum()
            total_dropped += simbad_dropped
            if simbad_dropped:
                logger.info("synchronize_state iter %d: dropped %d orphaned SIMBAD objects", iteration, simbad_dropped)
                self.simbad_objects = self.simbad_objects.loc[keep_simbad].reset_index(drop=True)

            # Prune observations pointing to missing SIMBAD objects
            valid_simbad_now = set(self.simbad_objects["main_id"])
            for name, df in list(self.datasets.items()):
                keep_obs = df["simbad_main_id"].isin(valid_simbad_now)
                obs_dropped = (~keep_obs).sum()
                total_dropped += obs_dropped
                if obs_dropped:
                    logger.info("synchronize_state iter %d: dropped %d orphaned observations in dataset '%s'", iteration, obs_dropped, name)
                    self.datasets[name] = df.loc[keep_obs].reset_index(drop=True)

            # Deduplicate relationships
            before_dedup = len(self.relationships)
            self.relationships = self.relationships.drop_duplicates().reset_index(drop=True)
            deduped = before_dedup - len(self.relationships)
            if deduped:
                logger.info("synchronize_state iter %d: removed %d duplicate edges", iteration, deduped)

            if total_dropped == 0:
                break
                
            iteration += 1

    def save(self, directory: str | Path) -> Path:
        """Persist all tables to Parquet files in directory.

        Returns the resolved directory path.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.simbad_objects.to_parquet(directory / "simbad_objects.parquet", index=False, engine="pyarrow")
        self.ads_papers.to_parquet(directory / "ads_papers.parquet", index=False, engine="pyarrow")
        self.relationships.to_parquet(directory / "relationships.parquet", index=False, engine="pyarrow")

        datasets_dir = directory / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        for name, df in self.datasets.items():
            df.to_parquet(datasets_dir / f"{name}.parquet", index=False, engine="pyarrow")

        logger.info("CrossmatchBundle saved to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CrossmatchBundle":
        """Load a previously-saved bundle from *directory*."""
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {directory}")

        simbad_path = directory / "simbad_objects.parquet"
        ads_path = directory / "ads_papers.parquet"
        rels_path = directory / "relationships.parquet"
        
        for p in (simbad_path, ads_path, rels_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file '{p.name}' in {directory}")

        simbad_objects = pd.read_parquet(simbad_path, engine="pyarrow")
        ads_papers = pd.read_parquet(ads_path, engine="pyarrow")
        relationships = pd.read_parquet(rels_path, engine="pyarrow")

        datasets = {}
        datasets_dir = directory / "datasets"
        if datasets_dir.exists() and datasets_dir.is_dir():
            for p in datasets_dir.glob("*.parquet"):
                datasets[p.stem] = pd.read_parquet(p, engine="pyarrow")

        return cls(
            datasets=datasets,
            simbad_objects=simbad_objects,
            ads_papers=ads_papers,
            relationships=relationships
        )

    def summary(self) -> str:
        """Human-readable snapshot of the bundle."""
        lines = ["CrossmatchBundle Summary", "=" * 40]
        lines.append(f"  simbad_objects: {len(self.simbad_objects)} rows")
        lines.append(f"  ads_papers: {len(self.ads_papers)} rows")
        lines.append(f"  relationships: {len(self.relationships)} edges")
        lines.append(f"  datasets ({len(self.datasets)} total):")
        for name, df in self.datasets.items():
            lines.append(f"    - {name}: {len(df)} observations")
        return "\n".join(lines)
