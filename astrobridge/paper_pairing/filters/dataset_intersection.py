from collections.abc import Sequence
import logging

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle
from astrobridge.paper_pairing.filters.base import BaseFilter
from astrobridge.paper_pairing.core.registry import DatasetSpec

logger = logging.getLogger(__name__)


class DatasetIntersectionFilter(BaseFilter):
    """
    Restricts the bundle to a specific set of datasets, and removes SIMBAD objects that 
    are not present in ALL of the requested datasets.
    """

    def __init__(self, datasets: Sequence[DatasetSpec]) -> None:
        self.datasets = datasets

    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        if not self.datasets:
            logger.warning("DatasetIntersectionFilter: No datasets requested. No changes made to the bundle.")
            return bundle

        for dataset in self.datasets:
            if dataset.name not in bundle.datasets.keys():
                raise ValueError(
                    f"Dataset '{dataset.name}' is required by the filter but is not present in the bundle."
                )

        required_dataset_names = {d.name for d in self.datasets}
        for dataset_name in list(bundle.datasets.keys()):
            if dataset_name not in required_dataset_names:
                logger.info("DatasetIntersectionFilter: Dropping dataset '%s'", dataset_name)
                del bundle.datasets[dataset_name]

        valid_simbad_ids = set(bundle.simbad_objects["main_id"])
        
        for dataset in self.datasets:
            dataset_simbad_ids = set(bundle.datasets[dataset.name]["simbad_main_id"].dropna())
            valid_simbad_ids = valid_simbad_ids.intersection(dataset_simbad_ids)

        keep_mask = bundle.simbad_objects["main_id"].isin(valid_simbad_ids)
        simbad_dropped = (~keep_mask).sum()
        
        if simbad_dropped:
            logger.info(
                "DatasetIntersectionFilter: Dropping %d SIMBAD objects missing from one or more required datasets.", 
                simbad_dropped
            )
            bundle.simbad_objects = bundle.simbad_objects.loc[keep_mask].reset_index(drop=True)

        return bundle
