import logging

import numpy as np
import pandas as pd

from astrobridge.paper_pairing.augmenters.alias import AliasAugmenter
from astrobridge.paper_pairing.core.bundle import CrossmatchBundle
from astrobridge.paper_pairing.filters.base import BaseFilter

logger = logging.getLogger(__name__)


class ObjectNameOrAliasInTitleOrAbstractFilter(BaseFilter):
    """
    Remove relationship rows where neither the astronomical object name (main_id)
    nor any of its aliases appear in the paper's title or abstract.
    """

    requires = [AliasAugmenter]

    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        rels = bundle.relationships

        merged = rels.merge(
            bundle.ads_papers[["bibcode", "paper_title", "abstract"]],
            on="bibcode",
            how="left",
        )

        merged = merged.merge(
            bundle.simbad_objects[["main_id", "aliases"]],
            left_on="simbad_main_id",
            right_on="main_id",
            how="left",
        )

        name_lower = merged["simbad_main_id"].fillna("").str.lower()
        title_lower = merged["paper_title"].fillna("").str.lower()
        abstract_lower = merged["abstract"].fillna("").str.lower()
        aliases_list = merged["aliases"].apply(
            lambda x: [str(a).lower() for a in x] if isinstance(x, list) else []
        )

        def is_match(name, aliases, title, abstract):
            names_to_check = [name] + aliases
            for n in names_to_check:
                if n and (n in title or n in abstract):
                    return True
            return False

        keep_mask = np.array(
            [
                is_match(name, aliases, title, abstract)
                for name, aliases, title, abstract in zip(
                    name_lower, aliases_list, title_lower, abstract_lower
                )
            ],
            dtype=bool,
        )

        before = len(rels)
        bundle.relationships = rels.loc[keep_mask].reset_index(drop=True)
        after = len(bundle.relationships)

        logger.info(
            "ObjectNameOrAliasInTitleOrAbstractFilter: kept %d / %d relationship edges",
            after,
            before,
        )

        return bundle
