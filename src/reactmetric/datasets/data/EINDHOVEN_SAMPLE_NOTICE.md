This directory contains a tiny schema-compatible fixture for the Eindhoven
Centraal loader and quickstart.

The full real dataset is not bundled because the Zenodo record is about 4.6 GB.
Use `reactmetric.datasets.load_eindhoven(download=True, days="01_10")` or pass a
local parquet path to load the real data.

Dataset attribution:

Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., & Toschi, F.
"Data-driven physics-based modeling of pedestrian dynamics - dataset:
Pedestrian trajectories at Eindhoven train station." Zenodo.
DOI: 10.5281/zenodo.13784588. License: CC-BY-4.0.
