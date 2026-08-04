from abc import ABC, abstractmethod

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle


class BaseFilter(ABC):
    """
    Base class for all filters.

    Rules
    --------
    - Filters may only remove rows — never add columns or rows.
    """

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
        if not inplace:
            bundle = bundle.copy()
        self._apply(bundle)
        bundle.synchronize_state()
        return bundle

    @abstractmethod
    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        """Filter the bundle in-place."""
        raise NotImplementedError()
