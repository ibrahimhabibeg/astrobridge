from abc import ABC, abstractmethod

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle


class BaseFilter(ABC):
    """
    Base class for all filters.

    Rules
    --------
    - Filters may only remove rows — never add columns or rows.
    """

    def apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        """Filter the bundle in-place and return it.

        Parameters
        ----------
        bundle : CrossmatchBundle
            The current pipeline state.

        Returns
        -------
        CrossmatchBundle
            The same bundle instance, with rows removed and state synchronized.
        """
        self._apply(bundle)
        bundle.synchronize_state()
        return bundle

    @abstractmethod
    def _apply(self, bundle: CrossmatchBundle) -> CrossmatchBundle:
        """Filter the bundle in-place."""
        raise NotImplementedError()
