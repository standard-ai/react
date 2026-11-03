# Datasets

Column/unit assumptions live in explicit **dataset mappings**:

```python
import reactmetric as rm

rm.datasets.mappings.eindhoven   # mm -> m, ms -> s, dt=0.1
rm.datasets.mappings.tracks      # Standard AI Tracks planar pose, dt=0.1
rm.datasets.mappings.ethucy      # whitespace frame/ped/x/y, dt=0.4

# Update a mapping (configuration surface — not the load path)
rm.datasets.specify_mapping("tracks", min_length=30)
rm.datasets.specify_mapping("my_store", path="store.yaml")  # load YAML into registry
rm.datasets.specify_mapping(
    "my_store",
    id="pid",
    coords=("x", "y"),
    time="t",
    dt=0.1,
    persist="store.yaml",
)
```

## Loading (default: unpartitioned)

```python
# Named defaults (mapping already registered)
data = rm.datasets.load_eindhoven_sample()           # the bundled 40-track excerpt
data = rm.datasets.load_eindhoven(download=True, days="01_10")

# Any local table through the same mapping-driven path
data = rm.datasets.load("/path/to/table.parquet", mapping="eindhoven")
data = rm.datasets.load("/path/to/tracks.csv", mapping="tracks")
data = rm.datasets.load("/path/to/mine.csv", mapping=rm.datasets.DatasetMapping.from_yaml("mine.yaml"))

# Synthetic / ETH/UCY
data, events = rm.datasets.synthetic_regime_changes(n=500, seed=0)
data = rm.datasets.load_eth_ucy("eth", root="/path/to/ethucy")
```

Calib vs evaluation for REACT is done **after load**:

```python
calib, test = data.split(fraction=0.3, seed=0)
```

## Optional on-disk `dataset_split` packaging

For large downloads or repeated demos, package a single source table into named
partition files. This is optional — loaders still accept a single file.

```python
result = rm.datasets.prepare(
    "/path/to/raw_tracks.parquet",
    mapping="tracks",
    output_dir="~/.cache/reactmetric/tracks_prepared",
    dataset_split=0.25,          # -> train + eval
    # dataset_split={"train": 0.7, "eval": 0.3},
    max_tracks=5000,             # optional subsample for huge corpora
)
data = rm.datasets.load_tracks(
    root=result.output_dir,
    dataset_split="eval",
)
```

`--dataset-split` / `dataset_split=` only selects a *named file* under a
directory. It is not the REACT calib/eval split.

## Eindhoven Centraal

Pedestrians crossing a platform of Eindhoven Centraal station, recorded at 10 Hz.

> Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., & Toschi, F. (2024). Data-driven
> physics-based modeling of pedestrian dynamics – dataset: Pedestrian trajectories at
> Eindhoven train station. Zenodo. <https://doi.org/10.5281/zenodo.13784588>
>
> Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., & Toschi, F. (2024). Data-driven
> physics-based modeling of pedestrian dynamics. *Physical Review E* 110(6), 064102.
> <https://doi.org/10.1103/PhysRevE.110.064102>

The full data are six 10-day parquet shards on Zenodo, each roughly 0.6 to 0.9 GB (days 01–10:
862 MB). `load_eindhoven(download=True, days="01_10")` fetches one into
`~/.cache/reactmetric/eindhoven/` with a progress line, once; `max_tracks=N, seed=0` then loads
a random subset of the tracks long enough to use (`min_length`).

Licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Please cite both if you
use the data. `load_eindhoven_sample()` returns a 40-track excerpt shipped with the package
(tracks selected, values unmodified); the pretrained GRU baselines were trained on this data.

Mapping: `object_identifier`, `time_ms`, `x_position_mm`, `y_position_mm` with
mm→m and ms→s scales (`rm.datasets.mappings.eindhoven`).

## Standard AI Tracks

Mapping: `person_track_id`, datetime `timestamp`, planar
`middle_of_waist_x` / `middle_of_waist_y` at 10 Hz
(`rm.datasets.mappings.tracks`). Default path: load a **single local**
CSV/parquet (e.g. the public Google Drive sample). Optionally
`prepare_tracks(...)` / `prepare(...)` to create `train`/`eval` files for
large sources.

```python
data = rm.datasets.load_tracks(path="/path/to/tracks_sample.csv")
calib, test = data.split(fraction=0.3, seed=0)
```

## Method-first philosophy

Built-in mappings ship with the package. Full datasets are loaded from local /
cached / authenticated sources; we redistribute adapters and tiny samples only.
