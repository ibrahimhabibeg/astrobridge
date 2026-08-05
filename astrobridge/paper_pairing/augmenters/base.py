from abc import ABC, abstractmethod

from astrobridge.paper_pairing.core.bundle import CrossmatchBundle


class BaseAugmenter(ABC):
    """Base class for all augmenters"""

    provides: list[tuple[str, str]] = []
    """(table_name, column_name) pairs this augmenter adds"""

    requires: list[type["BaseAugmenter"]] = []
    """Augmenter classes that must be applied before this one"""

    @classmethod
    def is_applied(cls, bundle: CrossmatchBundle) -> bool:
        """Return True if all columns declared in provides exist on the bundle"""
        if not cls.provides:
            return False
        for table_name, column_name in cls.provides:
            table = getattr(bundle, table_name, None)
            if table is None or column_name not in table.columns:
                return False
        return True

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
        self._check_requirements(bundle)
        if not inplace:
            bundle = bundle.copy()
        self._augment(bundle)
        return bundle

    @abstractmethod
    def _augment(self, bundle: CrossmatchBundle) -> None:
        """Augment the bundle in-place. Subclasses implement this."""
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


def inspect_augmentations(
    bundle: CrossmatchBundle,
    augmenters: list[type[BaseAugmenter]],
) -> dict[str, bool]:
    """
    Check which augmentations have been applied to a bundle.

    Parameters
    ----------
    bundle : CrossmatchBundle
        The bundle to inspect.
    augmenters : list[type[BaseAugmenter]]
        Augmenter classes to check.

    Returns
    -------
    dict[str, bool]
        Mapping of augmenter class name to whether it has been applied.
    """
    return {aug.__name__: aug.is_applied(bundle) for aug in augmenters}
