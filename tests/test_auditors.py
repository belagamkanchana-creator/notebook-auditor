from notebook_auditor.auditors.figure_auditor import FigureAuditor
from notebook_auditor.auditors.risk_auditor import RiskAuditor
from notebook_auditor.auditors.evidence_auditor import extract_observations
from notebook_auditor.claim_extractor import ClaimExtractor
from notebook_auditor.code_analysis import CodeIndex
from notebook_auditor.parser import load_notebook

from conftest import make_nb


def risks_for(cells, tmp_path, claims_md=""):
    nb = load_notebook(make_nb(cells, tmp_path))
    claims = ClaimExtractor().from_markdown(claims_md, "s.md") if claims_md else []
    return RiskAuditor(nb, CodeIndex(nb), extract_observations(nb), claims).run()


def checks(findings):
    return {f.check for f in findings}


def test_scaler_before_split_is_leakage(tmp_path):
    f = risks_for(
        [
            ("code", "scaler = StandardScaler()\nX_s = scaler.fit_transform(X)", []),
            ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, random_state=0)", []),
        ],
        tmp_path,
    )
    leak = [x for x in f if x.check == "data_leakage"]
    assert leak and leak[0].severity == "high" and leak[0].cell_indices == [1, 2]


def test_pipeline_after_split_is_clean(tmp_path):
    f = risks_for(
        [
            ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)", []),
            ("code", "scaler = StandardScaler()\nX_tr = scaler.fit_transform(X_tr)\nX_te = scaler.transform(X_te)", []),
        ],
        tmp_path,
    )
    assert "data_leakage" not in checks(f) and "test_set_contamination" not in checks(f)


def test_fit_on_test_is_contamination(tmp_path):
    f = risks_for(
        [
            ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)", []),
            ("code", "sc = MinMaxScaler()\nX_te2 = sc.fit_transform(X_te)", []),
        ],
        tmp_path,
    )
    assert "test_set_contamination" in checks(f)


def test_group_ids_without_group_split(tmp_path):
    f = risks_for([("code", "X_tr, X_te = train_test_split(df[['doc_id', 'text']], random_state=1)", [])], tmp_path)
    assert "group_leakage" in checks(f)


def test_missing_baseline_and_borderline(tmp_path):
    f = risks_for(
        [("code", "print(f'Accuracy: {acc:.3f}')\nprint(f'p-value: {p:.3f}')", ["Accuracy: 0.912\np-value: 0.052\n"])],
        tmp_path,
    )
    assert "missing_baseline" in checks(f)
    border = [x for x in f if x.check == "borderline_threshold"]
    assert border and "0.052" in border[0].message


def test_baseline_present_suppresses_flag(tmp_path):
    f = risks_for(
        [("code", "dummy = DummyClassifier(strategy='most_frequent').fit(X, y)\nprint('Accuracy:', acc)", ["Accuracy: 0.91\n"])],
        tmp_path,
    )
    assert "missing_baseline" not in checks(f)


def test_metric_threshold_borderline(tmp_path):
    f = risks_for([("code", "THRESHOLD = 0.80\nprint('F1:', f1)", ["F1: 0.803\n"])], tmp_path)
    assert any(x.check == "borderline_threshold" and "0.803" in x.message for x in f)
    f2 = risks_for([("code", "THRESHOLD = 0.80\nprint('F1:', f1)", ["F1: 0.83\n"])], tmp_path)
    assert "borderline_threshold" not in checks(f2)


def test_figure_checks(tmp_path):
    nb = load_notebook(
        make_nb(
            [
                ("code", "loss = [0.9, 0.7, 0.75, 0.5]\nplt.plot(loss, label='training loss')\nplt.show()", []),
                ("code", "plt.bar(['a', 'b'], [0.81, 0.83])\nplt.ylim(0.8, 0.84)\nplt.ylabel('Accuracy')\nplt.xlabel('Model')", []),
            ],
            tmp_path,
        )
    )
    claims = ClaimExtractor().from_markdown("- Training loss decreased steadily.\n", "s.md")
    fa = FigureAuditor(nb, CodeIndex(nb), claims)
    found = checks(fa.run())
    assert {"missing_axis_labels", "figure_not_rendered", "truncated_axis", "plot_trend_mismatch"} <= found
    assert fa.series[0].values == [0.9, 0.7, 0.75, 0.5] and fa.series[0].direction == "decrease"


def test_text_table_columns_label_numbers(tmp_path):
    from notebook_auditor.auditors.evidence_auditor import observations_for_cell

    out = " max_depth  train_acc  test_acc\n         1     0.5922    0.5946\n         2     0.6960    0.7004\n\nbest depth: 2\n"
    nb = load_notebook(make_nb([("code", "print(sweep.to_string(index=False))", [out])], tmp_path))
    labels = {(o.raw, o.label) for o in observations_for_cell(nb.cells[0])}
    assert ("0.7004", "test_acc") in labels and ("0.6960", "train_acc") in labels
    assert ("2", "best depth") in labels


def test_near_chance_ignores_integers_and_uses_label_counts(tmp_path):
    cells = [
        ("code", "df['label'].value_counts()", [{"output_type": "execute_result", "data": {"text/plain": "pos    70\nneg    30\nName: label, dtype: int64"}}]),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)\nprint(f'Test accuracy: {a}')\nDummyClassifier()", ["Test accuracy: 0.71\n"]),
    ]
    f = risks_for(cells, tmp_path)
    near = [x for x in f if x.check == "near_chance"]
    assert len(near) == 1 and "0.700" in near[0].message
