import logging
from typing import Sequence

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle
from astrobridge.paper_pairing.filters.base import BaseFilter

logger = logging.getLogger(__name__)


class TitleKeywordFilter(BaseFilter):
    """
    Filter ads_papers by keyword presence or title

    Parameters
    ----------
    keywords : Sequence[str]
        One or more keywords.  A paper is removed if it contains any of the keywords
    """

    def __init__(self, keywords: Sequence[str] = ['catalog', 'data release', 'survey', 'population', 'archive']
    ) -> None:
        if not keywords:
            raise ValueError("At least one keyword is required.")
        self.keywords = [kw.lower() for kw in keywords]

    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        before = len(bundle.ads_papers)
        mask = bundle.ads_papers['paper_title'].str.contains('|'.join(self.keywords), case=False, na=False)
        bundle.ads_papers = bundle.ads_papers[~mask].reset_index(drop=True)
        after = len(bundle.ads_papers)
        logger.info(
            "TitleKeywordFilter(keywords=%s): kept %d / %d papers",
            self.keywords,
            after,
            before,
        )

        return bundle
