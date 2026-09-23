from pose_av.runlog import flatten_metrics, log_remote, write_run


def test_flatten_nested_oks():
    flat = flatten_metrics(
        {
            "imgsz": 1280,
            "full_frame_bottom_up": {"mean_OKS": 0.9, "matched_instances": 53},
            "segment": "abc",
        }
    )
    assert flat["imgsz"] == 1280.0
    assert flat["full_frame_bottom_up/mean_OKS"] == 0.9
    assert "segment" not in flat


def test_write_run(tmp_path):
    p = write_run(tmp_path / "r", {"mean_OKS": 1.0}, {"seed": 42})
    assert p.exists()
    assert (tmp_path / "r" / "config.yaml").exists() or (tmp_path / "r" / "config.json").exists()


def test_log_remote_skips_without_keys(tmp_path):
    urls = log_remote({"mean_OKS": 1.0}, {"seed": 1}, {"wandb": {"enabled": True}, "mlflow": {"enabled": False}}, tmp_path)
    assert urls["wandb"] is None
    assert urls["mlflow"] is None
