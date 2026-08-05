"""
Builds a CrossmatchBundle
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

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

load_dotenv()

logger = logging.getLogger(__name__)


class CrossmatchFactory:
    """Build a :class:`CrossmatchBundle` from a Hugging Face dataset.

    Parameters
    ----------
    hf_dataset : str
        HF dataset path, e.g. ``"UniverseTBD/mmu_desi_edr_sv3"``.
    ads_token : str | None
        NASA ADS API bearer token.  Falls back to the ``ADS_API_TOKEN``
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
        hf_dataset: str,
        ads_token: str | None = None,
        simbad_batch_size: int = 200_000,
        simbad_radius: float = 0.1,
        tap_chunk_size: int = 20_000,
        ads_chunk_size: int = 2_000,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.ads_token = self._resolve_ads_token(ads_token)
        self.simbad_batch_size = simbad_batch_size
        self.simbad_radius = simbad_radius
        self.tap_chunk_size = tap_chunk_size
        self.ads_chunk_size = ads_chunk_size


    def build(self, save_dir: str | Path) -> CrossmatchBundle:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Stage 1/4: Ingesting HF dataset '%s'", self.hf_dataset)
        hf_objects = self._ingest_hf_catalog()
        logger.info("  -> %d unique HF objects", len(hf_objects))

        logger.info("Stage 2/4: SIMBAD crossmatch (batch=%d, radius=%.1f\")",
                     self.simbad_batch_size, self.simbad_radius)
        simbad_objects, hf_simbad_rels = self._simbad_crossmatch(hf_objects)
        logger.info("  -> %d SIMBAD objects, %d HF↔SIMBAD edges",
                     len(simbad_objects), len(hf_simbad_rels))


        logger.info("Stage 3/4: SIMBAD TAP -> bibcode mapping (chunk=%d)",
                     self.tap_chunk_size)
        simbad_bibcode_map = self._tap_bibcode_mapping(simbad_objects)
        logger.info("  -> %d (main_id, bibcode) pairs", len(simbad_bibcode_map))

        relationships = self._merge_relationships(hf_simbad_rels, simbad_bibcode_map)
        logger.info("  -> %d full relationship edges", len(relationships))

        unique_bibcodes = relationships["bibcode"].dropna().unique()
        logger.info("Stage 4/4: Fetching ADS metadata for %d unique bibcodes",
                     len(unique_bibcodes))
        ads_papers = self._fetch_ads_metadata(unique_bibcodes)
        logger.info("  -> %d ADS paper records", len(ads_papers))

        bundle = CrossmatchBundle(
            hf_objects=hf_objects,
            simbad_objects=simbad_objects,
            ads_papers=ads_papers,
            relationships=relationships,
        )
        bundle.synchronize_state()
        bundle.save(save_dir)
        logger.info("Pipeline complete.  Bundle saved to %s", save_dir)
        return bundle

    def _ingest_hf_catalog(self) -> pd.DataFrame:
        """Load and deduplicate the HF catalog."""
        catalog = lsdb.open_catalog(
            f"hf://datasets/{self.hf_dataset}", columns=["object_id"]
        )
        catalog_df = catalog.compute().to_pandas()
        return catalog_df[["object_id", "ra", "dec"]].reset_index(drop=True)

    def _simbad_crossmatch(
        self, hf_objects: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Batch-query SIMBAD and spatially match results.

        Returns
        -------
        simbad_objects : pd.DataFrame
            Unique SIMBAD entities (``main_id``, ``coo_bibcode``).
        relationships : pd.DataFrame
            Edge table with ``hf_id`` and ``simbad_main_id`` columns
            (``bibcode`` is left as NaN — filled in Stage 3).
        """
        coords = SkyCoord(
            ra=hf_objects["ra"].to_numpy(dtype=float) * u.deg,
            dec=hf_objects["dec"].to_numpy(dtype=float) * u.deg,
            frame="icrs",
        )
        search_radius = self.simbad_radius * u.arcsec

        all_matches: list[pd.DataFrame] = []

        for start in tqdm(
            range(0, len(hf_objects), self.simbad_batch_size),
            desc="SIMBAD batches",
        ):
            end = start + self.simbad_batch_size
            batch_coords = coords[start:end]

            result_table = Simbad.query_region(batch_coords, radius=search_radius)

            batch_df = hf_objects.iloc[start:end].reset_index(drop=True)
            batch_merged = self._merge_batch(
                batch_df,
                result_table.to_pandas() if result_table is not None else None,
                search_radius,
            )
            if not batch_merged.empty:
                all_matches.append(batch_merged)

            if start + self.simbad_batch_size < len(hf_objects):
                time.sleep(30)

        if not all_matches:
            merged = pd.DataFrame(
                columns=["main_id", "coo_bibcode", "object_id", "ra", "dec"]
            )
        else:
            merged = pd.concat(all_matches, ignore_index=True)

        simbad_objects = (
            merged[["main_id", "coo_bibcode"]]
            .drop_duplicates(subset="main_id")
            .reset_index(drop=True)
        )

        relationships = pd.DataFrame(
            {
                "hf_id": merged["object_id"],
                "simbad_main_id": merged["main_id"],
            }
        )

        return simbad_objects, relationships

    @staticmethod
    def _merge_batch(
        batch_df: pd.DataFrame,
        results_df: pd.DataFrame | None,
        search_radius: u.Quantity,
    ) -> pd.DataFrame:
        if results_df is None or results_df.empty:
            return pd.DataFrame(
                columns=["main_id", "coo_bibcode", "object_id", "ra", "dec"]
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
        out = batch_df.loc[valid, ["object_id", "ra", "dec"]].reset_index(drop=True).copy()
        out["main_id"] = matched_results.loc[valid, "main_id"].astype(str).str.strip().values
        out["coo_bibcode"] = matched_results.loc[valid, "coo_bibcode"].values

        return out[["main_id", "coo_bibcode", "object_id", "ra", "dec"]]

    def _tap_bibcode_mapping(self, simbad_objects: pd.DataFrame) -> pd.DataFrame:
        SIMBAD_TAP_URL = "http://simbad.u-strasbg.fr/simbad/sim-tap"
        service = vo.dal.TAPService(SIMBAD_TAP_URL)

        adql_query = """
        SELECT
            u.main_id,
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
                    if "main_id" in result_df.columns:
                        result_df["main_id"] = result_df["main_id"].astype(str).str.strip()
                    all_rows.append(result_df)
            except Exception:
                logger.exception("Error in SIMBAD TAP chunk")

            time.sleep(10)

        if not all_rows:
            return pd.DataFrame(columns=["main_id", "bibcode"])
        return pd.concat(all_rows, ignore_index=True)

    @staticmethod
    def _merge_relationships(
        hf_simbad: pd.DataFrame, simbad_bibcode: pd.DataFrame
    ) -> pd.DataFrame:
        merged = hf_simbad.merge(
            simbad_bibcode,
            left_on="simbad_main_id",
            right_on="main_id",
            how="inner",
        )
        return merged[["hf_id", "simbad_main_id", "bibcode"]].reset_index(drop=True)

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
        """Resolve ADS token from argument or environment."""
        if token:
            return token
        env_token = os.environ.get("ADS_API_TOKEN")
        if env_token:
            return env_token
        logger.error(
            "NASA ADS API token not found. Set ADS_API_TOKEN or pass ads_token=."
        )
        sys.exit(1)
