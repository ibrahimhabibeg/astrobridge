import logging

import numpy as np
import pandas as pd

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle
from astrobridge.paper_pairing.filters.base import BaseFilter

logger = logging.getLogger(__name__)


class ObjectNameInTitleOrAbstractFilter(BaseFilter):
    """
    Remove relationship rows where the astronomical object name (main_id) does not
    appear in the paper's title or abstract.
    """

    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        rels = bundle.relationships

        merged = rels.merge(
            bundle.ads_papers[["bibcode", "paper_title", "abstract"]],
            on="bibcode",
            how="left",
        )

        name_lower = merged["simbad_main_id"].fillna("").str.lower()
        title_lower = merged["paper_title"].fillna("").str.lower()
        abstract_lower = merged["abstract"].fillna("").str.lower()

        in_title = np.array(
            [
                name in title
                for name, title in zip(name_lower, title_lower)
            ],
            dtype=bool,
        )
        in_abstract = np.array(
            [
                name in abstract
                for name, abstract in zip(name_lower, abstract_lower)
            ],
            dtype=bool,
        )

        keep_mask = in_title | in_abstract

        before = len(rels)
        bundle.relationships = rels.loc[keep_mask].reset_index(drop=True)
        after = len(bundle.relationships)

        logger.info(
            "ObjectNameFilter: kept %d / %d relationship edges",
            after,
            before,
        )

        return bundle
