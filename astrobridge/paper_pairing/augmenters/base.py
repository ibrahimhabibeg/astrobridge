from abc import ABC, abstractmethod

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle


class BaseAugmenter(ABC):
    """Base class for all augmenters"""

    def augment(
        self, bundle: CrossmatchBundle, inplace: bool = True
    ) -> CrossmatchBundle:
        """Augment the bundle and return it.

        Parameters
        ----------
        bundle : CrossmatchBundle
            The current pipeline state.
        inplace : bool, optional
            If True (default), mutate bundle directly.  If False, operate
            on a deep copy and leave the original unchanged.

        Returns
        -------
        CrossmatchBundle
            The augmented bundle (same instance if inplace, a new copy otherwise).
        """
        if not inplace:
            bundle = bundle.copy()
        self._augment(bundle)
        return bundle

    @abstractmethod
    def _augment(self, bundle: CrossmatchBundle) -> None:
        """Augment the bundle in-place. Subclasses implement this."""
        raise NotImplementedError()
