import logging

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle
from astrobridge.paper_pairing.filters.base import BaseFilter

logger = logging.getLogger(__name__)


class MaxPapersPerObjectFilter(BaseFilter):

    def __init__(self, max_papers: int = 3) -> None:
        if max_papers < 1:
            raise ValueError("max_papers must be at least 1.")
        self.max_papers = max_papers

    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        rels = bundle.relationships

        paper_counts = rels.groupby("simbad_main_id")["bibcode"].nunique()
        over_limit = paper_counts[paper_counts > self.max_papers].index

        before = len(rels)
        bundle.relationships = rels[~rels["simbad_main_id"].isin(over_limit)].reset_index(drop=True)
        after = len(bundle.relationships)

        logger.info(
            "MaxPapersPerObjectFilter(max_papers=%d): removed %d objects, kept %d / %d edges",
            self.max_papers,
            len(over_limit),
            after,
            before,
        )

        return bundle


class MaxObjectsPerPaperFilter(BaseFilter):

    def __init__(self, max_objects: int = 3) -> None:
        if max_objects < 1:
            raise ValueError("max_objects must be at least 1.")
        self.max_objects = max_objects

    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        rels = bundle.relationships

        object_counts = rels.groupby("bibcode")["simbad_main_id"].nunique()
        over_limit = object_counts[object_counts > self.max_objects].index

        before = len(rels)
        bundle.relationships = rels[~rels["bibcode"].isin(over_limit)].reset_index(drop=True)
        after = len(bundle.relationships)

        logger.info(
            "MaxObjectsPerPaperFilter(max_objects=%d): removed %d papers, kept %d / %d edges",
            self.max_objects,
            len(over_limit),
            after,
            before,
        )

        return bundle
