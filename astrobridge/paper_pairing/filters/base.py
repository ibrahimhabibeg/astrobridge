from abc import ABC, abstractmethod

from astrobridge.paper_pairing.augmenters.base import BaseAugmenter
from astrobridge.paper_pairing.core.bundle import CrossmatchBundle


class BaseFilter(ABC):
    """Base class for all filters."""

    requires: list[type[BaseAugmenter]] = []
    """Augmenter classes that must be applied before this filter can run."""

    def apply(self, bundle: CrossmatchBundle, inplace: bool = True) -> CrossmatchBundle:
        """Filter the bundle and return it.

        Parameters
        ----------
        bundle : CrossmatchBundle
            The current pipeline state.
        inplace : bool, optional
            If True (default), mutate *bundle* directly. If False, operate on a
            deep copy and leave the original unchanged.

        Returns
        -------
        CrossmatchBundle
            The filtered bundle (same instance if inplace, a new copy otherwise).
        """
        self._check_requirements(bundle)
        if not inplace:
            bundle = bundle.copy()
        self._apply(bundle)
        bundle.synchronize_state()
        return bundle

    @abstractmethod
    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        """Filter the bundle in-place."""
        raise NotImplementedError()

    def _check_requirements(self, bundle: CrossmatchBundle) -> None:
        """Raise if any required augmentations have not been applied."""
        for req in self.requires:
            if not req.is_applied(bundle):
                missing_cols = [
                    f"{table}.{col}"
                    for table, col in req.provides
                    if col not in getattr(bundle, table, {})
                ]
                raise RuntimeError(
                    f"{self.__class__.__name__} requires {req.__name__}, "
                    f"but it has not been applied (missing: {missing_cols})"
                )
