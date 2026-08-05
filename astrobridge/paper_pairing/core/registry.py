from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a Hugging Face dataset to ingest.

    Parameters
    ----------
    name : str
        The internal name for this dataset in the bundle (e.g., "desi_edr").
    hf_path : str
        The Hugging Face dataset path (e.g., "UniverseTBD/mmu_desi_edr_sv3").
    id_col : str, optional
        The column name containing unique observation IDs (default: "object_id").
    ra_col : str, optional
        The column name containing Right Ascension (default: "ra").
    dec_col : str, optional
        The column name containing Declination (default: "dec").
    """

    name: str
    hf_path: str
    id_col: str = "object_id"
    ra_col: str = "ra"
    dec_col: str = "dec"


DESI_SPECTRA = DatasetSpec(
    name="desi_spectra", hf_path="UniverseTBD/mmu_desi_edr_sv3"
)

SDSS_SPECTRA = DatasetSpec(
    name="sdss_spectra", hf_path="UniverseTBD/mmu_sdss_sdss"
)

PLASTICC = DatasetSpec(
    name="plasticc", hf_path="UniverseTBD/mmu_plasticc"
)

CHANDRA = DatasetSpec(
    name="chandra", hf_path="UniverseTBD/mmu_chandra_spectra"
)

LEGACY_SURVEY_SOUTH = DatasetSpec(
    name="legacy_survey_south", hf_path="hugging-science/mmu_legacysurvey_dr10_south_21"
)

LEGACY_SURVEY_NORTH = DatasetSpec(
    name="legacy_survey_north", hf_path="UniverseTBD/mmu_ssl_legacysurvey_north"
)

BUILTIN_DATASETS = {
    "desi_spectra": DESI_SPECTRA,
    "sdss": SDSS_SPECTRA,
    "plasticc": PLASTICC,
    "chandra": CHANDRA,
    "legacy_survey_south": LEGACY_SURVEY_SOUTH,
    "legacy_survey_north": LEGACY_SURVEY_NORTH,
}
