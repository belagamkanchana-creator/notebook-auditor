import json

from notebook_auditor.cli import main
from notebook_auditor.llm import LLMClient, parse_json_response
from notebook_auditor.pipeline import run_audit

from conftest import make_nb


def verdicts(result):
    return {result.claim(v.claim_id).text: v for v in result.verdicts}


def test_sample_audit_end_to_end(sample_nb_path, slides_path):
    r = run_audit(sample_nb_path, [slides_path], provider="none")
    v = verdicts(r)
    assert v["We collected 3,020 documents from an online review site"].verdict == "CONTRADICTED"
    assert v["We used a stratified 80/20 train/test split"].verdict == "CONTRADICTED"
    assert v["Features were standardized after splitting to avoid leakage"].verdict == "CONTRADICTED"
    assert v["Validation loss decreased steadily across all 6 epochs"].verdict == "CONTRADICTED"
    assert v["Our model achieves 85% accuracy on the held-out test set"].verdict == "METHODOLOGICAL_RISK"
    assert v["The improvement over SVM is statistically significant (p = 0.048)"].verdict == "METHODOLOGICAL_RISK"
    assert v["The dataset is imbalanced: 252 reviews are positive"].verdict == "SUPPORTED"
    tsne = v["The t-SNE plot demonstrates that the model clearly separates positive and negative reviews"]
    assert tsne.verdict == "UNVERIFIABLE" and tsne.strength == "INTERPRETATION"
    assert "statistical separation" in tsne.does_not_prove
    found = {f.check for f in r.risks}
    assert {"numeric_discrepancy", "data_leakage", "missing_baseline", "borderline_threshold", "truncated_axis", "overclaiming"} <= found
    # evidence points at the exact cell
    c1 = v["We collected 3,020 documents from an online review site"]
    assert c1.evidence[0].cell_index == 3 and "420" in c1.evidence[0].snippet


def test_cli_writes_reports(sample_nb_path, slides_path, tmp_path):
    code = main(["run", "--notebook", sample_nb_path, "--slides", slides_path, "--output-dir", str(tmp_path), "--provider", "none", "-q", "--fail-on", "contradicted"])
    assert code == 1  # contradicted claims exist
    md = (tmp_path / "audit_report.md").read_text()
    data = json.loads((tmp_path / "audit_report.json").read_text())
    assert "## Claim Verdicts" in md and '<a id="cell-3"></a>' in md
    assert data["summary"]["verdicts"]["CONTRADICTED"] >= 4
    assert {"claims", "plans", "executions", "verdicts", "risks", "evidence_map"} <= set(data)


def test_notebook_markdown_claims_when_no_slides(tmp_path):
    path = make_nb(
        [
            ("code", "print(f'Accuracy: {acc:.3f}')", ["Accuracy: 0.912\n"]),
            ("markdown", "The classifier reaches an accuracy of 91.2%.", []),
        ],
        tmp_path,
    )
    r = run_audit(path, [], provider="none")
    assert len(r.claims) == 1 and r.claims[0].source == "notebook:Cell 2"
    assert r.verdicts[0].verdict == "SUPPORTED"


class FakeLLM(LLMClient):
    """Scripted LLM: hallucinates on purpose to test that agent output is validated."""

    provider = "fake"

    def __init__(self):
        super().__init__("fake-1")

    def complete(self, system, user, max_tokens=2000):
        if "PLANNER" in system:
            return json.dumps({"plans": [{"claim_id": "C1", "target_cells": [1, 99], "rationale": "prints metrics"}]})
        if "EXECUTOR" in system:
            return json.dumps({"values": [
                {"cell_index": 1, "location": "output", "quote": "Recall: 0.77", "value": 0.77, "label": "recall"},  # real
                {"cell_index": 1, "location": "output", "quote": "Precision: 0.93", "value": 0.93, "label": "precision"},  # invented
            ]})
        if "VERIFIER" in system:
            return json.dumps({"verdict": "SUPPORTED", "confidence": 0.99, "rationale": "looks right", "citations": [{"cell_index": 1, "quote": "not in the cell"}]})
        return json.dumps({"claims": [{"unit_id": 0, "claim": "The model has a precision of 0.93", "category": "numeric"},
                                      {"unit_id": 0, "claim": "The model has a recall of 0.77", "category": "numeric"}]})


def test_llm_outputs_are_validated(tmp_path):
    nb = make_nb([("code", "print(report)", ["Recall: 0.77\nRuntime: 12s\n"])], tmp_path)
    slides = tmp_path / "s.md"
    slides.write_text("- The model has a recall of 0.77 and a precision of 0.93.\n")
    r = run_audit(nb, [str(slides)], llm=FakeLLM())
    v = verdicts(r)
    assert r.plans[0].target_cells == [1]  # out-of-range cell 99 dropped
    assert v["The model has a recall of 0.77"].verdict == "SUPPORTED"
    prec = v["The model has a precision of 0.93"]
    assert prec.verdict == "UNVERIFIABLE"  # invented quote rejected, uncited LLM verdict discarded
    assert any("Rejected LLM value" in w for w in r.warnings)
    assert r.mode == "llm:fake/fake-1"


def test_llm_claims_with_new_numbers_are_dropped(tmp_path):
    class Hallucinator(FakeLLM):
        def complete(self, system, user, max_tokens=2000):
            if "extract atomic" in system:
                return json.dumps({"claims": [{"unit_id": 0, "claim": "Accuracy was 0.99", "category": "numeric"}]})
            return "{}"

    nb = make_nb([("code", "print(acc)", ["Accuracy: 0.85\n"])], tmp_path)
    slides = tmp_path / "s.md"
    slides.write_text("- Accuracy was 0.85.\n")
    r = run_audit(nb, [str(slides)], llm=Hallucinator())
    assert [c.text for c in r.claims] == ["Accuracy was 0.85"]  # fell back to heuristic claims


def test_parse_json_response():
    assert parse_json_response('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response("nothing here") is None


def test_failing_llm_is_disabled_and_reported(sample_nb_path, slides_path):
    class Down(LLMClient):
        provider = "down"

        def __init__(self):
            super().__init__("m")

        def complete(self, system, user, max_tokens=2000):
            raise ConnectionError("unreachable")

    llm = Down()
    r = run_audit(sample_nb_path, [slides_path], llm=llm)
    assert llm.calls == 2 and llm.disabled  # stops after two consecutive failures
    assert r.mode.startswith("heuristic (LLM down/m unavailable")
    assert any("disabled after 2" in w for w in r.warnings)
    assert r.summary()["verdicts"]["CONTRADICTED"] >= 4  # heuristic verdicts still produced
