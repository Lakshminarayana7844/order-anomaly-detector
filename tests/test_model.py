import numpy as np

from src.features import FeatureBuilder
from src.model import AnomalyModel


def _fitted(small_df):
    builder = FeatureBuilder().fit(small_df)
    feats = builder.transform(small_df)
    model = AnomalyModel().fit(feats)
    return builder, feats, model


def test_scores_in_range(small_df):
    _, feats, model = _fitted(small_df)
    scored = model.score(feats)
    assert scored["anomaly_score"].between(0, 1.5).all()
    assert scored["is_anomaly"].dtype == bool


def test_injected_anomalies_score_higher(small_df):
    _, feats, model = _fitted(small_df)
    scored = model.score(feats)
    mean_anom = scored.loc[small_df["is_anomaly"], "anomaly_score"].mean()
    mean_norm = scored.loc[~small_df["is_anomaly"], "anomaly_score"].mean()
    assert mean_anom > mean_norm * 1.5


def test_recall_reasonable(small_df):
    _, feats, model = _fitted(small_df)
    scored = model.score(feats)
    recall = (scored["is_anomaly"] & small_df["is_anomaly"]).sum() / small_df["is_anomaly"].sum()
    assert recall >= 0.6


def test_save_load_roundtrip(small_df, tmp_path):
    _, feats, model = _fitted(small_df)
    path = tmp_path / "m.pkl"
    model.save(str(path))
    loaded = AnomalyModel.load(str(path))
    a = model.score(feats.head(20))["anomaly_score"].to_numpy()
    b = loaded.score(feats.head(20))["anomaly_score"].to_numpy()
    assert np.allclose(a, b)
