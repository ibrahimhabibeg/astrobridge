"""
Builds a CrossmatchBundle from multiple Hugging Face MMU datasets.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from collections.abc import Sequence

import astropy.units as u
import lsdb
import pandas as pd
import pyvo as vo
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad
from dotenv import load_dotenv
from tqdm.auto import tqdm

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle
from astrobridge.paper_pairing.core.registry import DatasetSpec, BUILTIN_DATASETS

load_dotenv()

logger = logging.getLogger(__name__)

class CrossmatchFactory:
    """Build a :class:`CrossmatchBundle` from Hugging Face dataset(s).

    Parameters
    ----------
    datasets : Sequence[str | DatasetSpec]
        A list of datasets to process. Can be the name of a built-in dataset
        or a custom DatasetSpec object.
    ads_token : str | None
        NASA ADS API bearer token. Falls back to the ``ADS_API_TOKEN``
        environment variable when ``None``.
    simbad_batch_size : int
        Coordinates per SIMBAD ``query_region`` batch.
    simbad_radius : float
        Crossmatch radius in arcseconds.
    tap_chunk_size : int
        Unique IDs per SIMBAD TAP upload chunk.
    ads_chunk_size : int
        Bibcodes per ADS BigQuery request.
    """

    def __init__(
        self,
        datasets: Sequence[str | DatasetSpec],
        ads_token: str | None = None,
        simbad_batch_size: int = 200_000,
        simbad_radius: float = 0.1,
        tap_chunk_size: int = 20_000,
        ads_chunk_size: int = 2_000,
    ) -> None:
        self.datasets = self._resolve_datasets(datasets)
        self.ads_token = self._resolve_ads_token(ads_token)
        self.simbad_batch_size = simbad_batch_size
        self.simbad_radius = simbad_radius
        self.tap_chunk_size = tap_chunk_size
        self.ads_chunk_size = ads_chunk_size

    @staticmethod
    def _resolve_datasets(datasets: Sequence[str | DatasetSpec]) -> list[DatasetSpec]:
        resolved = []
        for ds in datasets:
            if isinstance(ds, str):
                if ds not in BUILTIN_DATASETS:
                    raise ValueError(f"Unknown built-in dataset name: '{ds}'")
                resolved.append(BUILTIN_DATASETS[ds])
            elif isinstance(ds, DatasetSpec):
                resolved.append(ds)
            else:
                raise TypeError(f"Invalid dataset type: {type(ds)}")
        return resolved

    def build(self, save_dir: str | Path) -> CrossmatchBundle:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Stage 1/4 & 2/4: Ingesting and cross-matching %d datasets", len(self.datasets))
        
        bundle_datasets: dict[str, pd.DataFrame] = {}
        all_simbad_matches: list[pd.DataFrame] = []

        for spec in self.datasets:
            logger.info("  -> Processing dataset: %s", spec.name)
            obs_df = self._ingest_hf_catalog(spec)
            logger.info("     Found %d unique HF objects", len(obs_df))

            matched_obs_df, simbad_df = self._simbad_crossmatch(obs_df, spec)
            
            bundle_datasets[spec.name] = matched_obs_df
            all_simbad_matches.append(simbad_df)
            
            logger.info("     %d cross-matched to SIMBAD", len(matched_obs_df))

        # Combine all unique SIMBAD objects from all datasets
        if not all_simbad_matches:
            logger.warning("No SIMBAD matches found across any dataset!")
            simbad_objects = pd.DataFrame(columns=["main_id", "ra", "dec", "coo_bibcode"])
        else:
            simbad_objects = pd.concat(all_simbad_matches, ignore_index=True)
            simbad_objects = simbad_objects.drop_duplicates(subset=["main_id"]).reset_index(drop=True)
            
        logger.info("  -> Pooled %d unique SIMBAD objects across all datasets", len(simbad_objects))

        logger.info("Stage 3/4: SIMBAD TAP -> bibcode mapping (chunk=%d)", self.tap_chunk_size)
        relationships = self._tap_bibcode_mapping(simbad_objects)
        logger.info("  -> %d full relationship edges (main_id <-> bibcode)", len(relationships))

        unique_bibcodes = relationships["bibcode"].dropna().unique()
        logger.info("Stage 4/4: Fetching ADS metadata for %d unique bibcodes", len(unique_bibcodes))
        ads_papers = self._fetch_ads_metadata(unique_bibcodes)
        logger.info("  -> %d ADS paper records", len(ads_papers))

        bundle = CrossmatchBundle(
            datasets=bundle_datasets,
            simbad_objects=simbad_objects,
            ads_papers=ads_papers,
            relationships=relationships,
        )
        
        logger.info("Synchronizing state...")
        bundle.synchronize_state()
        
        bundle.save(save_dir)
        logger.info("Pipeline complete. Bundle saved to %s", save_dir)
        return bundle

    @classmethod
    def append_dataset(cls, bundle: CrossmatchBundle, dataset: str | DatasetSpec, simbad_radius: float = 0.1) -> None:
        """
        Appends a new HF dataset to an existing CrossmatchBundle.
        
        Only observations matching existing SIMBAD objects in the bundle are kept.
        """
        resolved = cls._resolve_datasets([dataset])[0]
        logger.info("Appending dataset '%s' to existing bundle", resolved.name)
        
        if resolved.name in bundle.datasets:
            logger.warning("Dataset '%s' already exists in the bundle. Overwriting.", resolved.name)

        logger.info("  -> Loading HF dataset %s", resolved.hf_path)
        hf_cat = lsdb.open_catalog(
            f"hf://datasets/{resolved.hf_path}", 
            columns=[resolved.id_col, resolved.ra_col, resolved.dec_col]
        )
        
        logger.info("  -> Preparing SIMBAD reference catalog (%d objects)", len(bundle.simbad_objects))
        
        simbad_df = bundle.simbad_objects[["main_id", "ra", "dec"]].rename(
            columns={"ra": "simbad_ra", "dec": "simbad_dec"}
        )
        simbad_cat = lsdb.from_dataframe(
            simbad_df,
            ra_column="simbad_ra",
            dec_column="simbad_dec",
            margin_threshold=simbad_radius
        )
        
        logger.info("  -> Executing spatial crossmatch (radius=%.1f\")", simbad_radius)
        matched = hf_cat.crossmatch(
            simbad_cat, 
            radius_arcsec=simbad_radius,
            suffix_method="overlapping_columns"
        ).compute().to_pandas()
        
        if matched.empty:
            logger.warning("No matches found for dataset '%s'.", resolved.name)
            matched_obs_df = pd.DataFrame(columns=["object_id", "ra", "dec", "simbad_main_id"])
        else:
            rename_map = {resolved.id_col: "object_id"}
            matched = matched.rename(columns=rename_map)
            
            matched_obs_df = matched[["object_id", "ra", "dec", "main_id"]].copy()
            matched_obs_df = matched_obs_df.rename(columns={"main_id": "simbad_main_id"})
            
            matched_obs_df["simbad_main_id"] = matched_obs_df["simbad_main_id"].astype(str).str.strip()
            matched_obs_df = matched_obs_df.drop_duplicates(subset=["object_id"]).reset_index(drop=True)

        logger.info("  -> Added %d new observations from '%s'", len(matched_obs_df), resolved.name)
        
        bundle.datasets[resolved.name] = matched_obs_df
        
        logger.info("  -> Synchronizing bundle state")
        bundle.synchronize_state()

    def _ingest_hf_catalog(self, spec: DatasetSpec) -> pd.DataFrame:
        catalog = lsdb.open_catalog(
            f"hf://datasets/{spec.hf_path}", columns=[spec.id_col, spec.ra_col, spec.dec_col]
        )
        catalog_df = catalog.compute().to_pandas()
        
        # Standardize columns
        rename_map = {
            spec.id_col: "object_id",
            spec.ra_col: "ra",
            spec.dec_col: "dec"
        }
        df = catalog_df.rename(columns=rename_map)
        return df[["object_id", "ra", "dec"]].reset_index(drop=True)

    def _simbad_crossmatch(
        self, obs_df: pd.DataFrame, spec: DatasetSpec
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Batch-query SIMBAD and spatially match results for a single dataset.

        Returns
        -------
        matched_obs_df : pd.DataFrame
            The observation dataframe with `simbad_main_id` appended.
        simbad_objects : pd.DataFrame
            The unique SIMBAD entities (``main_id``, ``ra``, ``dec``, ``coo_bibcode``) matched from this dataset.
        """
        coords = SkyCoord(
            ra=obs_df["ra"].to_numpy(dtype=float) * u.deg,
            dec=obs_df["dec"].to_numpy(dtype=float) * u.deg,
            frame="icrs",
        )
        search_radius = self.simbad_radius * u.arcsec

        all_matches: list[pd.DataFrame] = []

        for start in tqdm(
            range(0, len(obs_df), self.simbad_batch_size),
            desc=f"SIMBAD batches ({spec.name})",
        ):
            end = start + self.simbad_batch_size
            batch_coords = coords[start:end]

            result_table = Simbad.query_region(batch_coords, radius=search_radius)

            batch_df = obs_df.iloc[start:end].reset_index(drop=True)
            batch_merged = self._merge_batch(
                batch_df,
                result_table.to_pandas() if result_table is not None else None,
                search_radius,
            )
            if not batch_merged.empty:
                all_matches.append(batch_merged)

            if start + self.simbad_batch_size < len(obs_df):
                time.sleep(30)

        if not all_matches:
            merged = pd.DataFrame(
                columns=["main_id", "simbad_ra", "simbad_dec", "coo_bibcode", "object_id", "obs_ra", "obs_dec"]
            )
        else:
            merged = pd.concat(all_matches, ignore_index=True)

        simbad_objects = (
            merged[["main_id", "simbad_ra", "simbad_dec", "coo_bibcode"]]
            .rename(columns={"simbad_ra": "ra", "simbad_dec": "dec"})
            .drop_duplicates(subset=["main_id"])
            .reset_index(drop=True)
        )

        matched_obs_df = (
            merged[["object_id", "obs_ra", "obs_dec", "main_id"]]
            .rename(columns={"obs_ra": "ra", "obs_dec": "dec", "main_id": "simbad_main_id"})
            .reset_index(drop=True)
        )

        return matched_obs_df, simbad_objects

    @staticmethod
    def _merge_batch(
        batch_df: pd.DataFrame,
        results_df: pd.DataFrame | None,
        search_radius: u.Quantity,
    ) -> pd.DataFrame:
        if results_df is None or results_df.empty:
            return pd.DataFrame(
                columns=["main_id", "simbad_ra", "simbad_dec", "coo_bibcode", "object_id", "obs_ra", "obs_dec"]
            )

        batch_coords = SkyCoord(
            ra=batch_df["ra"].astype(float).to_numpy() * u.deg,
            dec=batch_df["dec"].astype(float).to_numpy() * u.deg,
            frame="icrs",
        )
        result_coords = SkyCoord(
            ra=results_df["ra"].astype(float).to_numpy() * u.deg,
            dec=results_df["dec"].astype(float).to_numpy() * u.deg,
            frame="icrs",
        )

        idx, d2d, _ = batch_coords.match_to_catalog_sky(result_coords)
        valid = d2d <= search_radius

        matched_results = results_df.iloc[idx].reset_index(drop=True)
        
        out = pd.DataFrame({
            "object_id": batch_df.loc[valid, "object_id"].values,
            "obs_ra": batch_df.loc[valid, "ra"].values,
            "obs_dec": batch_df.loc[valid, "dec"].values,
            "main_id": matched_results.loc[valid, "main_id"].astype(str).str.strip().values,
            "simbad_ra": matched_results.loc[valid, "ra"].values,
            "simbad_dec": matched_results.loc[valid, "dec"].values,
            "coo_bibcode": matched_results.loc[valid, "coo_bibcode"].values,
        })

        return out

    def _tap_bibcode_mapping(self, simbad_objects: pd.DataFrame) -> pd.DataFrame:
        SIMBAD_TAP_URL = "http://simbad.u-strasbg.fr/simbad/sim-tap"
        service = vo.dal.TAPService(SIMBAD_TAP_URL)

        adql_query = """
        SELECT
            u.main_id AS simbad_main_id,
            rf.bibcode
        FROM TAP_UPLOAD.input_table AS u, basic AS b, has_ref AS hr, ref AS rf
        WHERE u.main_id = b.main_id
          AND b.oid = hr.oidref
          AND hr.oidbibref = rf.oidbib
        """

        unique_ids = simbad_objects["main_id"].dropna().drop_duplicates()
        chunks = [
            unique_ids.iloc[i : i + self.tap_chunk_size]
            for i in range(0, len(unique_ids), self.tap_chunk_size)
        ]

        all_rows: list[pd.DataFrame] = []

        for chunk in tqdm(chunks, desc="SIMBAD TAP chunks"):
            upload_table = Table.from_pandas(pd.DataFrame({"main_id": chunk}))
            try:
                job = service.run_sync(
                    adql_query,
                    uploads={"input_table": upload_table},
                    maxrec=3_000_000,
                )
                result_df = job.to_table().to_pandas()
                if not result_df.empty:
                    if "simbad_main_id" in result_df.columns:
                        result_df["simbad_main_id"] = result_df["simbad_main_id"].astype(str).str.strip()
                    all_rows.append(result_df)
            except Exception:
                logger.exception("Error in SIMBAD TAP chunk")

            time.sleep(10)

        if not all_rows:
            return pd.DataFrame(columns=["simbad_main_id", "bibcode"])
        return pd.concat(all_rows, ignore_index=True)

    def _fetch_ads_metadata(self, unique_bibcodes) -> pd.DataFrame:
        url = "https://api.adsabs.harvard.edu/v1/search/bigquery"
        headers = {
            "Authorization": f"Bearer {self.ads_token}",
            "Content-Type": "big-query/csv",
        }

        ads_rows: list[dict] = []

        chunks = [
            unique_bibcodes[i : i + self.ads_chunk_size]
            for i in range(0, len(unique_bibcodes), self.ads_chunk_size)
        ]

        for chunk in tqdm(chunks, desc="ADS BigQuery chunks"):
            payload = "bibcode\n" + "\n".join(chunk)
            params = {"q": "*:*", "fl": "bibcode,title,abstract,doi,keyword", "rows": 2000}

            try:
                resp = requests.post(
                    url, headers=headers, params=params, data=payload
                )
                resp.raise_for_status()

                docs = resp.json().get("response", {}).get("docs", [])
                for paper in docs:
                    bibcode = paper.get("bibcode")
                    ads_rows.append(
                        {
                            "bibcode": bibcode,
                            "paper_title": paper.get("title", ["No Title Available"])[0],
                            "abstract": paper.get("abstract", "No Abstract Available"),
                            "doi": paper.get("doi", []),
                            "keyword": paper.get("keyword", []),
                            "preprint_url": (
                                f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
                            ),
                        }
                    )
            except Exception:
                logger.exception("Error in ADS BigQuery chunk")

            time.sleep(5)

        if not ads_rows:
            return pd.DataFrame(
                columns=["bibcode", "paper_title", "abstract", "doi", "preprint_url"]
            )
        return pd.DataFrame(ads_rows)

    @staticmethod
    def _resolve_ads_token(token: str | None) -> str:
        if token:
            return token
        env_token = os.environ.get("ADS_API_TOKEN")
        if env_token:
            return env_token
        logger.error(
            "NASA ADS API token not found. Set ADS_API_TOKEN or pass ads_token=."
        )
        sys.exit(1)
