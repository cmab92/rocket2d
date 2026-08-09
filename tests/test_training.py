from rocket2d.training import run_cnn, run_rocket


def test_run_cnn_smoke(neu_dataset_dir):
    metrics = run_cnn(
        "neu",
        str(neu_dataset_dir),
        img_size=16,
        batch_size=4,
        epochs=1,
        seed=0,
        device="cpu",
        show_plots=False,
    )
    assert "accuracy" in metrics


def test_run_rocket_smoke(neu_dataset_dir):
    metrics = run_rocket(
        "neu",
        str(neu_dataset_dir),
        img_size=20,
        seed=0,
        n_kernels=20,
        show_plots=False,
    )
    assert "accuracy" in metrics
