import numpy as np
import pytest

import reactmetric as rm


@pytest.fixture(autouse=True)
def _reset_mappings():
    rm.datasets.reset_mappings()
    yield
    rm.datasets.reset_mappings()


def test_synthetic_has_events_and_tracks():
    data, events = rm.datasets.synthetic_regime_changes(n=10, seed=0)
    assert len(data) == 10
    assert all(t.D == 2 for t in data)
    assert sum(len(v) for v in events.values()) > 0


def test_eindhoven_sample_loads_and_has_schema():
    ts = rm.datasets.load_eindhoven_sample()
    assert len(ts) > 10
    assert all(t.D == 2 for t in ts)
    assert "10.5281/zenodo.13784588" in rm.datasets.eindhoven_citation()
    assert rm.datasets.mappings.eindhoven.id == "object_identifier"
    assert rm.datasets.mappings.eindhoven.position_scale == 1e-3


def test_load_eindhoven_without_cache_or_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=True") as err:
        rm.datasets.load_eindhoven(cache_dir=str(tmp_path), download=False)
    assert str(tmp_path) in str(err.value) and "load_eindhoven_sample" in str(err.value)
    with pytest.raises(FileNotFoundError):
        rm.datasets.load_dataset("eindhoven", cache_dir=str(tmp_path))


def test_unified_load_uses_eindhoven_mapping(tmp_path):
    # Write a tiny Eindhoven-schema CSV and load via mapping name.
    path = tmp_path / "tiny.csv"
    path.write_text(
        "time_ms,object_identifier,x_position_mm,y_position_mm\n"
        + "\n".join(
            f"{t * 100},{pid},{1000 + t},{2000 - t}"
            for pid in (1, 2)
            for t in range(25)
        )
        + "\n"
    )
    ts = rm.datasets.load(str(path), mapping="eindhoven")
    assert len(ts) == 2
    np.testing.assert_allclose(ts[1].positions[0], [1.0, 2.0])
    assert ts.dt == pytest.approx(0.1)


def test_specify_mapping_updates_registry_and_yaml(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    m = rm.datasets.specify_mapping(
        "custom",
        id="pid",
        coords=("x", "y"),
        time="t",
        dt=0.2,
        persist=str(yaml_path),
    )
    assert rm.datasets.mappings.custom.dt == 0.2
    assert yaml_path.exists()
    loaded = rm.datasets.DatasetMapping.from_yaml(str(yaml_path))
    assert loaded.id == "pid"
    assert m.name == "custom"

    rm.datasets.specify_mapping("custom", min_length=50)
    assert rm.datasets.mappings.custom.min_length == 50


def test_tracks_loads_single_local_csv(tmp_path):
    path = tmp_path / "tracks.csv"
    rows = ["person_track_id,timestamp,middle_of_waist_x,middle_of_waist_y"]
    for tid in (10, 11):
        for i in range(30):
            rows.append(f"{tid},{i * 0.1},{0.1 * i},{0.2 * i}")
    path.write_text("\n".join(rows) + "\n")
    ts = rm.datasets.load_tracks(path=str(path))
    assert len(ts) == 2
    assert ts.dt == pytest.approx(0.1)
    calib, test = ts.split(fraction=0.5, seed=0)
    assert len(calib) + len(test) == 2


def test_tracks_dataset_split_selects_named_file(tmp_path):
    root = tmp_path / "dir"
    root.mkdir()
    (root / "eval.csv").write_text(
        "person_track_id,timestamp,middle_of_waist_x,middle_of_waist_y\n"
        + "\n".join(f"1,{i * 0.1},{i},0" for i in range(25))
        + "\n"
    )
    (root / "train.csv").write_text(
        "person_track_id,timestamp,middle_of_waist_x,middle_of_waist_y\n"
        + "\n".join(f"2,{i * 0.1},{i},1" for i in range(25))
        + "\n"
    )
    ts = rm.datasets.load_tracks(root=str(root), dataset_split="eval")
    assert list(ts.ids) == [1]


def test_tracks_requires_local_path():
    with pytest.raises(FileNotFoundError, match="local file"):
        rm.datasets.load_tracks()


def test_prepare_creates_train_eval_partitions(tmp_path):
    source = tmp_path / "all.csv"
    rows = ["person_track_id,timestamp,middle_of_waist_x,middle_of_waist_y"]
    for tid in range(1, 21):
        for i in range(25):
            rows.append(f"{tid},{i * 0.1},{0.1 * i},{tid}")
    source.write_text("\n".join(rows) + "\n")
    out = tmp_path / "prepared"
    result = rm.datasets.prepare(
        str(source),
        mapping="tracks",
        output_dir=str(out),
        dataset_split=0.25,
        seed=0,
        fmt="csv",
    )
    assert set(result.paths) == {"train", "eval"}
    assert sum(result.n_tracks.values()) == 20
    assert result.n_tracks["eval"] == 5
    assert (out / "eval.csv").exists()
    assert (out / "train.csv").exists()

    eval_ts = rm.datasets.load_tracks(root=str(out), dataset_split="eval")
    assert len(eval_ts) == 5

    # Unpartitioned load still works on the original source.
    all_ts = rm.datasets.load_tracks(path=str(source))
    assert len(all_ts) == 20


def test_prepare_tracks_alias_and_max_tracks(tmp_path):
    source = tmp_path / "all.csv"
    rows = ["person_track_id,timestamp,middle_of_waist_x,middle_of_waist_y"]
    for tid in range(1, 31):
        for i in range(25):
            rows.append(f"{tid},{i * 0.1},{i},0")
    source.write_text("\n".join(rows) + "\n")
    out = tmp_path / "prepared"
    result = rm.datasets.prepare_tracks(
        str(source),
        str(out),
        dataset_split={"train": 0.5, "eval": 0.5},
        max_tracks=10,
        seed=1,
        fmt="csv",
    )
    assert sum(result.n_tracks.values()) == 10


def test_eth_ucy_adapter_reads_frame_ped_x_y(tmp_path):
    rows = []
    for ped in range(3):
        for frame in range(30):
            rows.append([frame, ped, 0.1 * frame + ped, -0.05 * frame])
    path = tmp_path / "scene.txt"
    np.savetxt(path, np.array(rows))
    ts = rm.datasets.load_eth_ucy(str(path), min_length=20, dt=0.4)
    assert len(ts) == 3
    assert ts.dt == 0.4
    assert ts[0].D == 2


def test_eth_ucy_adapter_reads_raw_obsmat_x_z_y(tmp_path):
    rows = []
    for ped in range(2):
        for frame in range(30):
            rows.append([frame, ped, 0.1 * frame, 999.0, -0.2 * frame, 0.0, 0.0, 0.0])
    path = tmp_path / "obsmat.txt"
    np.savetxt(path, np.array(rows))
    ts = rm.datasets.load_eth_ucy(str(path), min_length=20, dt=0.4)
    assert len(ts) == 2
    np.testing.assert_allclose(ts[0].positions[-1], [2.9, -5.8])


def test_datasets_exports():
    assert hasattr(rm.datasets, "synthetic_regime_changes")
    assert hasattr(rm.datasets, "load_eindhoven")
    assert hasattr(rm.datasets, "load_eindhoven_sample")
    assert hasattr(rm.datasets, "download_eindhoven")
    assert hasattr(rm.datasets, "load_eth_ucy")
    assert hasattr(rm.datasets, "load_tracks")
    assert hasattr(rm.datasets, "prepare_tracks")
    assert hasattr(rm.datasets, "load")
    assert hasattr(rm.datasets, "load_dataset")
    assert hasattr(rm.datasets, "prepare")
    assert hasattr(rm.datasets, "create_dataset_split")
    assert hasattr(rm.datasets, "mappings")
    assert hasattr(rm.datasets, "specify_mapping")
    assert hasattr(rm.datasets, "DatasetMapping")
