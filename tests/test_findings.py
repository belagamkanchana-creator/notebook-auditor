import json

from notebook_auditor.llm import LLMClient
from notebook_auditor.pipeline import run_audit

from conftest import make_nb


def texts(result):
    return [f.text for f in result.findings]


def test_sample_findings_and_gaps(sample_nb_path):
    r = run_audit(sample_nb_path, [], provider="none")  # no write-up -> findings mode on by default
    t = texts(r)
    assert "Data size: Loaded 420 documents" in t
    assert "The largest class makes up 60.0% of the labels" in t
    acc = next(f for f in r.findings if f.text == "Test accuracy is 0.8452")
    assert acc.trust == "at_risk" and acc.cells == [7]
    assert any("before the train/test split" in c for c in acc.caveats)
    loss = next(f for f in r.findings if f.text.startswith("Validation loss falls"))
    assert "not steadily" in loss.text and loss.basis == "CODE"
    assert sum(1 for x in t if x.startswith("Statistical test")) == 1  # the t statistic is not a p-value
    gaps = [g.title for g in r.gaps]
    assert gaps[0] == "Fit preprocessing on the training data only"
    assert "Add a baseline" in gaps and "Report uncertainty" in gaps
    assert r.summary()["findings"] == len(r.findings)


def test_findings_off_when_checking_a_writeup(sample_nb_path, slides_path):
    r = run_audit(sample_nb_path, [slides_path], provider="none")
    assert r.findings == [] and r.claims


def test_overfitting_found_in_depth_sweep(tmp_path):
    table = (" max_depth  train_acc  test_acc\n"
             "         2     0.70    0.70\n         4     0.79    0.80\n         8     0.87    0.82\n"
             "        12     0.94    0.80\n        15     0.98    0.78\n\nbest depth: 8 (test accuracy 0.82)\n")
    nb = make_nb([("code", "print(sweep.to_string(index=False))", [table])], tmp_path)
    r = run_audit(nb, [], provider="none")
    over = [f for f in r.findings if f.kind == "overfitting"]
    assert over and "max_depth = 8" in over[0].text and "0.98" in over[0].text
    test_trend = next(f for f in r.findings if f.text.startswith("Test accuracy rises"))
    assert test_trend.text.endswith("best 0.82 at max_depth = 8") and "to 0.78" in test_trend.text


def test_comparison_uses_row_names_not_index(tmp_path):
    html = ("<table><tr><th></th><th>model</th><th>test_accuracy</th></tr>"
            "<tr><th>0</th><td>majority baseline</td><td>0.34</td></tr>"
            "<tr><th>1</th><td>random forest</td><td>0.85</td></tr></table>")
    nb = make_nb([("code", "comparison", [{"output_type": "execute_result", "data": {"text/plain": "...", "text/html": html}}])], tmp_path)
    r = run_audit(nb, [], provider="none")
    assert any(f.text == "Test_accuracy: best for random forest (0.85), worst for majority baseline (0.34)" for f in r.findings)


def test_probabilities_are_not_p_values(tmp_path):
    nb = make_nb([("code", "print(rates)", ["Japanese | P(→L2 | in L1) = 4.2%\nHindi | P(→L2 | in L1) = 24.8%\n"])], tmp_path)
    r = run_audit(nb, [], provider="none")
    assert not any(f.kind == "significance" for f in r.findings)


class FindingsLLM(LLMClient):
    provider = "fake"

    def __init__(self):
        super().__init__("vision-1")
        self.images_seen = 0

    def complete_vision(self, system, user, images, max_tokens=2000):
        self.images_seen = len(images)
        return self.complete(system, user, max_tokens)

    def complete(self, system, user, max_tokens=2000):
        if "careful research assistant" in system:
            return json.dumps({"findings": [
                {"text": "Per-class recall is lowest for neutral at 0.41", "cell_index": 1, "quote": "neutral recall: 0.41", "kind": "comparison"},
                {"text": "Recall for neutral is 0.93", "cell_index": 1, "quote": "", "kind": "metric"},  # number not printed
                {"text": "The confusion matrix shows neutral is mostly confused with positive", "cell_index": 2, "quote": "", "kind": "figure"},
                {"text": "Accuracy is 0.77", "cell_index": 1, "quote": "accuracy 0.99 overall", "kind": "metric"},  # fake quote
            ]})
        return "{}"


def test_claude_findings_are_validated(tmp_path):
    from conftest import EXAMPLES  # noqa: F401  (ensures conftest import side effects)
    import importlib.util

    spec = importlib.util.spec_from_file_location("mk", EXAMPLES / "make_sample_notebook.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    fig = mk.figure()
    nb = make_nb(
        [("code", "print(report)", ["neutral recall: 0.41\npositive recall: 0.90\naccuracy 0.77\n"]),
         ("code", "ConfusionMatrixDisplay.from_predictions(y, p)\nplt.show()", [fig])],
        tmp_path,
    )
    llm = FindingsLLM()
    r = run_audit(nb, [], llm=llm, findings=True)
    t = texts(r)
    assert llm.images_seen == 1  # the figure was sent for reading
    assert "Per-class recall is lowest for neutral at 0.41" in t
    fig_finding = next(f for f in r.findings if f.kind == "figure")
    assert fig_finding.trust == "visual" and fig_finding.basis == "VISUAL"
    assert "Recall for neutral is 0.93" not in t  # number not printed in the cited cell
    assert not any("0.99" in x for x in t)
    assert any("numbers not printed" in w for w in r.warnings) and any("quote is not in Cell 1" in w for w in r.warnings)
