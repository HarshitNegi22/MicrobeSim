import numpy as np
import pandas as pd
import pytest

from src.ui import common


def test_benchmark_group_column_is_forwarded_to_grouped_validation(monkeypatch):
    observed = {}
    expected_result = object()

    def fake_load_or_train(frame, model_keys, **kwargs):
        observed.update(kwargs)
        return expected_result

    monkeypatch.setattr(common, "load_or_train", fake_load_or_train)
    frame = pd.DataFrame({"study_id": ["study-a", "study-b", "study-a"]})

    result = common._benchmark_dataset(frame, ("rf",), "study_id")

    assert result is expected_result
    assert observed["group_name"] == "study_id"
    np.testing.assert_array_equal(observed["groups"], ["study-a", "study-b", "study-a"])


def test_benchmark_group_column_rejects_single_group(monkeypatch):
    monkeypatch.setattr(common, "load_or_train", lambda *args, **kwargs: pytest.fail("trained"))
    frame = pd.DataFrame({"batch": ["only", "only"]})

    with pytest.raises(ValueError, match="at least two groups"):
        common._benchmark_dataset(frame, ("rf",), "batch")
