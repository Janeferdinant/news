"""Tests for the real classical-model service, using tiny models trained on a few invented articles."""
import joblib
import numpy as np
import pytest
from services import ClassicalModels, Registry, _top_terms

nltk = pytest.importorskip("nltk")


@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    from fakenews.features import clean_text, feature_names, full_feature_vector

    try:
        clean_text("probe")
    except LookupError:  # NLTK data missing and no network
        pytest.skip("NLTK data not available")

    calm = ["officials said the report would be reviewed on tuesday",
            "the council published figures and a spokesperson told reporters more detail would follow",
            "analysts quoted in the report said the effect would be modest"] * 8
    loud = ["shocking truth hiding cover up you will not believe it share now",
            "caught hiding the truth shocking coverup nobody is talking about this",
            "wake up they are lying shocking secret revealed"] * 8
    texts, y = calm + loud, [1] * len(calm) + [0] * len(loud)
    vec = TfidfVectorizer(max_features=60).fit([clean_text(t) for t in texts])
    X = np.vstack([full_feature_vector("", t, vec)[0] for t in texts])
    lr = LogisticRegression(max_iter=500, random_state=0).fit(X, y)
    selected = np.argsort(-np.abs(lr.coef_[0]))[:30]
    X_sel = X[:, selected]
    lr = LogisticRegression(max_iter=500, random_state=0).fit(X_sel, y)
    xgb = XGBClassifier(n_estimators=8, max_depth=2, random_state=0, eval_metric="logloss").fit(X_sel, y)

    d = tmp_path_factory.mktemp("models")
    joblib.dump(lr, d / "fake_news_lr_model.pkl")
    joblib.dump(xgb, d / "fake_news_xgb_model.pkl")
    joblib.dump(vec, d / "tfidf_vectorizer.pkl")
    joblib.dump(selected, d / "selected_indices.pkl")
    return d, lr, xgb, selected, feature_names(vec)[selected]


LOUD = "SHOCKING truth: they have been hiding the coverup!"
CALM = "Officials said the report would be reviewed on Tuesday."


def test_labels_and_probabilities_match_the_models(tiny_artifacts):
    d, lr, xgb, selected, _ = tiny_artifacts
    service = ClassicalModels(d)
    for title, text in (("", LOUD), ("", CALM)):
        out = service.predict_both(title, text)
        x = service._features(title, text)
        assert out["logistic_regression"]["label"] == ("real" if lr.predict(x)[0] == 1 else "fake")
        assert out["xgboost"]["label"] == ("real" if xgb.predict(x)[0] == 1 else "fake")
        p = lr.predict_proba(x)[0]
        assert out["logistic_regression"]["probabilities"]["fake"] == pytest.approx(p[0], abs=1e-4)
        assert out["logistic_regression"]["confidence"] == pytest.approx(max(p), abs=1e-4)


def test_loud_and_calm_examples_are_separated(tiny_artifacts):
    service = ClassicalModels(tiny_artifacts[0])
    assert service.predict_both("", LOUD)["logistic_regression"]["label"] == "fake"
    assert service.predict_both("", CALM)["logistic_regression"]["label"] == "real"


def test_lr_terms_explain_the_score(tiny_artifacts):
    d, lr, _, selected, names = tiny_artifacts
    service = ClassicalModels(d)
    out = service.predict_both("", LOUD)["logistic_regression"]
    x = service._features("", LOUD)[0]
    contrib = lr.coef_[0] * x
    assert contrib.sum() + lr.intercept_[0] == pytest.approx(lr.decision_function([x])[0])
    terms = out["top_terms"]
    assert 0 < len(terms) <= 8
    assert [abs(t["contribution"]) for t in terms] == sorted((abs(t["contribution"]) for t in terms), reverse=True)
    for t in terms:
        assert t["direction"] == ("real" if t["contribution"] > 0 else "fake")
        i = list(names).index(t["term"])
        assert t["contribution"] == pytest.approx(contrib[i], abs=1e-4)
    assert out["selected_features_total"] == 30
    assert out["selected_features_present"] == int(np.count_nonzero(x))


def test_xgb_terms_are_booster_contributions(tiny_artifacts):
    import xgboost as xgb_lib

    d, _, xgb, _, names = tiny_artifacts
    service = ClassicalModels(d)
    out = service.predict_both("", LOUD)["xgboost"]
    x = service._features("", LOUD)
    full = xgb.get_booster().predict(xgb_lib.DMatrix(x), pred_contribs=True)[0]
    assert full.sum() == pytest.approx(xgb.get_booster().predict(xgb_lib.DMatrix(x), output_margin=True)[0], abs=1e-4)
    assert out["top_terms"]
    for t in out["top_terms"]:
        i = list(names).index(t["term"])
        assert t["contribution"] == pytest.approx(full[i], abs=1e-4)
        assert x[0][i] != 0  # only terms that occur in the article are listed


def test_article_without_selected_terms_has_no_terms():
    assert _top_terms(np.zeros(3), ["a", "b", "c"], np.zeros(3)) == []


def test_absent_terms_are_not_listed_even_with_a_contribution():
    terms = _top_terms(np.array([5.0, -1.0, 0.5]), ["absent", "present", "also"], np.array([0.0, 0.3, 0.2]))
    assert [t["term"] for t in terms] == ["present", "also"]


def test_registry_from_env_reports_missing_artifacts_without_paths(tmp_path):
    reg = Registry.from_env({"NEWS_CLASSICAL_DIR": str(tmp_path / "nope"),
                             "NEWS_DISTILBERT_DIR": str(tmp_path / "nada")})
    assert reg.models == {}
    listing = {m["id"]: m for m in reg.listing()}
    assert not any(m["available"] for m in listing.values())
    assert str(tmp_path) not in " ".join(m["description"] for m in listing.values())


def test_registry_from_env_loads_classical_models(tiny_artifacts, tmp_path):
    reg = Registry.from_env({"NEWS_CLASSICAL_DIR": str(tiny_artifacts[0]),
                             "NEWS_DISTILBERT_DIR": str(tmp_path / "none")})
    assert set(reg.models) == {"logistic_regression", "xgboost"}
    assert reg.listing()[2]["available"] is False
    assert reg.models["xgboost"].predict("", LOUD)["model"] == "xgboost"
