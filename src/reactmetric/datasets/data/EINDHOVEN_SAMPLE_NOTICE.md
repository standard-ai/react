# Eindhoven Centraal sample

`eindhoven_sample.csv` is an excerpt of real data: 40 pedestrian tracks from the Eindhoven
Centraal platform dataset, day 2 of the 01–10 shard, drawn with seed 0 from tracks of 150 to
450 observations and spread across the day's 24 hours
(`scripts/export_eindhoven_csv.py --days day02 --tracks 40 --min-length 150 --max-length 450`).

Changes from the original: only the selection of tracks and of the four columns `time_ms`,
`object_identifier`, `x_position_mm` and `y_position_mm`. Values, ids and timestamps are
unmodified.

The sample is for trying the library. It is far too small for conclusions; load the full data
with `reactmetric.datasets.load_eindhoven(download=True)` (about 860 MB per 10-day shard).

## Attribution

Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., Toschi, F. "Data-driven
physics-based modeling of pedestrian dynamics – dataset: Pedestrian trajectories at Eindhoven
train station." Zenodo (2024). https://doi.org/10.5281/zenodo.13784588

Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., Toschi, F. "Data-driven
physics-based modeling of pedestrian dynamics." Physical Review E 110(6), 064102 (2024).
https://doi.org/10.1103/PhysRevE.110.064102

Licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0),
https://creativecommons.org/licenses/by/4.0/. The authors do not endorse this library.
