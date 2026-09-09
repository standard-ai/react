import matplotlib

import reactmetric as rm

matplotlib.use("Agg")


def test_viz_defaults_to_tgf_dark_theme(monkeypatch):
    monkeypatch.delenv("REACT_PLOT_THEME", raising=False)
    import matplotlib.pyplot as plt

    rm.viz.plot_calibration(
        rm.calibration.CalibrationReport(
            model_name="model",
            dim=2,
            n_samples=10,
            coverage_chi2=2.0,
            pit=(1.0, 1.0),
            sharpness=0.0,
            mean_bias=(0.0, 0.0),
        )
    )
    assert plt.rcParams["figure.facecolor"] == "#0B1622"
    assert plt.rcParams["axes.facecolor"] == "#0B1622"


def test_requested_quickstart_plots_write_files(synthetic, tmp_path):
    events = synthetic["events"]
    test = synthetic["test"]
    ref = synthetic["ref"]
    cv = rm.baselines.ConstantVelocity()
    accuracy = rm.forecast_accuracy({"ConstantVelocity": cv}, test, leads=(1, 5, 10))

    overview = tmp_path / "dataset_trajectory_overview.png"
    task_km = tmp_path / "task_react_km_curves.png"
    perf = tmp_path / "forecast_performance_summary.png"

    rm.viz.dataset_trajectory_overview(test, events, max_tracks=10, save=str(overview))
    rm.viz.task_react_km_curves(
        {"ConstantVelocity": cv},
        test,
        ref,
        events,
        lead=10,
        distortion_m=0.5,
        save=str(task_km),
    )
    rm.viz.forecast_performance_summary(
        accuracy,
        leads=(1, 5, 10),
        save=str(perf),
    )

    assert overview.exists()
    assert task_km.exists()
    assert perf.exists()
