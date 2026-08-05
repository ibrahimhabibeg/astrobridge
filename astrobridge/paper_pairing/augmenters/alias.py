"""
Augments simbad_objects with all known SIMBAD aliases
"""

import logging
import time

import pandas as pd
import pyvo as vo
from astropy.table import Table
from tqdm.auto import tqdm

from astrobridge.paper_pairing.augmenters.base import BaseAugmenter
from astrobridge.paper_pairing.core.bundle import CrossmatchBundle

logger = logging.getLogger(__name__)

SIMBAD_TAP_URL = "http://simbad.u-strasbg.fr/simbad/sim-tap"


class AliasAugmenter(BaseAugmenter):
    """Add an aliases column (list[str]) to simbad_objects.

    Parameters
    ----------
    tap_chunk_size : int
        Number of main_id values to upload per TAP request.
    sleep_seconds : float
        Seconds to sleep between TAP requests to respect rate limits.
    """

    provides = [("simbad_objects", "aliases")]

    def __init__(
        self, tap_chunk_size: int = 20_000, sleep_seconds: float = 10.0
    ) -> None:
        self.tap_chunk_size = tap_chunk_size
        self.sleep_seconds = sleep_seconds

    def _augment(self, bundle: CrossmatchBundle) -> None:
        unique_ids = bundle.simbad_objects["main_id"].dropna().drop_duplicates()
        logger.info(
            "AliasAugmenter: fetching aliases for %d SIMBAD objects", len(unique_ids)
        )

        raw_pairs = self._query_aliases(unique_ids)

        alias_map: dict[str, list[str]] = (
            raw_pairs.groupby("main_id")["alias"]
            .apply(list)
            .to_dict()
        )

        bundle.simbad_objects["aliases"] = (
            bundle.simbad_objects["main_id"]
            .map(alias_map)
            .apply(lambda v: v if isinstance(v, list) else [])
        )

        total_aliases = bundle.simbad_objects["aliases"].apply(len).sum()
        logger.info(
            "AliasAugmenter: added %d total aliases across %d objects",
            total_aliases,
            len(bundle.simbad_objects),
        )

    def _query_aliases(self, unique_ids: pd.Series) -> pd.DataFrame:
        """
        Query the SIMBAD TAP ident table in batches.

        Returns a DataFrame with columns main_id and alias.
        """
        service = vo.dal.TAPService(SIMBAD_TAP_URL)

        adql_query = """
        SELECT
            u.main_id,
            id.id AS alias
        FROM TAP_UPLOAD.input_table AS u
        JOIN basic AS b ON u.main_id = b.main_id
        JOIN ident AS id ON b.oid = id.oidref
        """

        chunks = [
            unique_ids.iloc[i : i + self.tap_chunk_size]
            for i in range(0, len(unique_ids), self.tap_chunk_size)
        ]

        all_rows: list[pd.DataFrame] = []

        for chunk in tqdm(chunks, desc="SIMBAD TAP alias chunks"):
            upload_table = Table.from_pandas(pd.DataFrame({"main_id": chunk}))
            try:
                job = service.run_sync(
                    adql_query,
                    uploads={"input_table": upload_table},
                    maxrec=5_000_000,
                )
                result_df = job.to_table().to_pandas()
                if not result_df.empty:
                    result_df["main_id"] = (
                        result_df["main_id"].astype(str).str.strip()
                    )
                    result_df["alias"] = (
                        result_df["alias"].astype(str).str.strip()
                    )
                    all_rows.append(result_df)
            except Exception:
                logger.exception("Error in SIMBAD TAP alias chunk")

            time.sleep(self.sleep_seconds)

        if not all_rows:
            return pd.DataFrame(columns=["main_id", "alias"])
        return pd.concat(all_rows, ignore_index=True)
