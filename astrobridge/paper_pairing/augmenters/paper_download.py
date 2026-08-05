"""
Downloads paper PDFs from ADS preprint URLs and tracks download status.

Adds two columns to ads_papers:
  - download_status:  "pending" | "success" | "error_transient" | "error_permanent"
  - download_message: None on success/pending, descriptive reason on failure
"""

import logging
import time

import requests
from tqdm.auto import tqdm

from astrobridge.assets.manager import AssetManager
from astrobridge.paper_pairing.augmenters.base import BaseAugmenter
from astrobridge.paper_pairing.core.bundle import CrossmatchBundle

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_ERROR_TRANSIENT = "error_transient"
STATUS_ERROR_PERMANENT = "error_permanent"

_PERMANENT_HTTP_CODES = {400, 403, 404, 410, 451}


class PaperDownloadAugmenter(BaseAugmenter):
    """Download paper PDFs and track status on ads_papers.

    Parameters
    ----------
    asset_manager : AssetManager
        File store for saving/checking PDFs.
    sleep_seconds : float
        Seconds to sleep between download attempts to respect rate limits.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        asset_manager: AssetManager,
        sleep_seconds: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        self.asset_manager = asset_manager
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout

    def _augment(self, bundle: CrossmatchBundle) -> None:
        papers = bundle.ads_papers

        if "download_status" not in papers.columns:
            papers["download_status"] = STATUS_PENDING
            papers["download_message"] = None

        success_mask = papers["download_status"] == STATUS_SUCCESS
        for idx in papers.index[success_mask]:
            bibcode = papers.at[idx, "bibcode"]
            if not self.asset_manager.is_available(bibcode):
                papers.at[idx, "download_status"] = STATUS_PENDING
                papers.at[idx, "download_message"] = None
                logger.info("File missing for '%s', reset to pending", bibcode)

        actionable = papers["download_status"].isin(
            [STATUS_PENDING, STATUS_ERROR_TRANSIENT]
        )
        to_download = papers.index[actionable]

        logger.info(
            "PaperDownloadAugmenter: %d papers to attempt (%d total, %d already done, "
            "%d permanent failures)",
            len(to_download),
            len(papers),
            (papers["download_status"] == STATUS_SUCCESS).sum(),
            (papers["download_status"] == STATUS_ERROR_PERMANENT).sum(),
        )

        for idx in tqdm(to_download, desc="Downloading papers"):
            bibcode = papers.at[idx, "bibcode"]
            url = papers.at[idx, "preprint_url"]

            status, message = self._download_one(bibcode, url)
            papers.at[idx, "download_status"] = status
            papers.at[idx, "download_message"] = message

            if status != STATUS_SUCCESS:
                logger.warning("Failed '%s': [%s] %s", bibcode, status, message)

            time.sleep(self.sleep_seconds)

        counts = papers["download_status"].value_counts().to_dict()
        logger.info("PaperDownloadAugmenter complete: %s", counts)

    def _download_one(self, bibcode: str, url: str) -> tuple[str, str | None]:
        """Attempt to download a single paper.

        Returns (status, message).
        """
        try:
            resp = requests.get(url, timeout=self.timeout, allow_redirects=True)
        except requests.ConnectionError as e:
            return STATUS_ERROR_TRANSIENT, f"ConnectionError: {e}"
        except requests.Timeout:
            return STATUS_ERROR_TRANSIENT, f"Timeout after {self.timeout}s"
        except requests.RequestException as e:
            return STATUS_ERROR_TRANSIENT, f"RequestException: {e}"

        if resp.status_code in _PERMANENT_HTTP_CODES:
            return STATUS_ERROR_PERMANENT, f"HTTP {resp.status_code}"

        if resp.status_code >= 500:
            return STATUS_ERROR_TRANSIENT, f"HTTP {resp.status_code}"

        if resp.status_code != 200:
            return STATUS_ERROR_TRANSIENT, f"HTTP {resp.status_code}"

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
            return STATUS_ERROR_PERMANENT, f"Not a PDF (Content-Type: {content_type})"
        self.asset_manager.save(bibcode, resp.content)
        return STATUS_SUCCESS, None
