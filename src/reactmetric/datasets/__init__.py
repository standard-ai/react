"""Dataset adapters, explicit mappings, and a synthetic regime-change generator."""

from .mapping import (
    DatasetMapping,
    MappingRegistry,
    mappings,
    reset_mappings,
    specify_mapping,
)
from .synthetic import synthetic_regime_changes

__all__ = [
    "synthetic_regime_changes",
    "DatasetMapping",
    "MappingRegistry",
    "mappings",
    "specify_mapping",
    "reset_mappings",
]

try:
    from .load import load, load_dataset

    __all__ += ["load", "load_dataset"]
except Exception:  # pragma: no cover
    pass

try:
    from .prepare import PrepareResult, create_dataset_split, prepare

    __all__ += ["prepare", "create_dataset_split", "PrepareResult"]
except Exception:  # pragma: no cover
    pass

try:
    from .eindhoven import (
        DOI as EINDHOVEN_DOI,
    )
    from .eindhoven import (
        LICENSE as EINDHOVEN_LICENSE,
    )
    from .eindhoven import (
        citation as eindhoven_citation,
    )
    from .eindhoven import (
        download_eindhoven,
        eindhoven_file_manifest,
        load_eindhoven,
        load_eindhoven_sample,
    )

    __all__ += [
        "load_eindhoven",
        "load_eindhoven_sample",
        "download_eindhoven",
        "eindhoven_file_manifest",
        "eindhoven_citation",
        "EINDHOVEN_DOI",
        "EINDHOVEN_LICENSE",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from .ethucy import load_eth_ucy as load_eth_ucy

    __all__.append("load_eth_ucy")
except Exception:  # pragma: no cover
    pass

try:
    from .tracks import load_tracks as load_tracks
    from .tracks import prepare_tracks as prepare_tracks

    __all__ += ["load_tracks", "prepare_tracks"]
except Exception:  # pragma: no cover
    pass
