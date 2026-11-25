from notebook_auditor.claim_extractor import ClaimExtractor
from notebook_auditor.models import Observation
from notebook_auditor.textnum import extract_numbers, series_direction, values_match


def obs(value, raw=None, pct=False, decimals=None):
    raw = raw or str(value)
    dec = decimals if decimals is not None else (len(raw.split(".")[1]) if "." in raw else 0)
    return Observation(1, "output", value, raw, pct, dec, "", [], [], raw)


def test_number_extraction_skips_model_names_and_refs():
    nums = extract_numbers("GPT-4 on Figure 2 reached 85.3% accuracy over 3,020 docs, F1 0.80", exclude_refs=True)
    assert [n.raw for n in nums] == ["85.3%", "3,020", "0.80"]
    assert nums[0].kinds == ["accuracy"] and nums[1].kinds == ["count"] and nums[2].kinds == ["f1"]
    assert nums[1].qualifier == "gt"


def test_tolerant_matching():
    claim = extract_numbers("85% accuracy")[0]
    assert values_match(claim, obs(0.8452))
    assert not values_match(claim, obs(0.8312))
    assert values_match(extract_numbers("F1 of 0.80")[0], obs(0.801))
    assert not values_match(extract_numbers("3,020 documents")[0], obs(420))
    assert values_match(extract_numbers("about 3,000 documents")[0], obs(3020))


def test_series_direction_tolerates_tiny_bumps():
    assert series_direction([0.69, 0.52, 0.44, 0.47, 0.41]) == ("decrease", False)
    assert series_direction([0.9, 0.8, 0.7]) == ("decrease", True)
    assert series_direction([0.5, 0.6, 0.7, 0.699, 0.8]) == ("increase", True)


def test_claims_are_atomic_resolved_and_categorised():
    md = """## Results
- Our model achieves 85% accuracy and the baseline scored 50%.
- It reaches a macro F1 of 0.80.
- We used a stratified 80/20 train/test split.
- Validation loss decreased steadily.
- The embeddings clearly capture sentiment.
- Questions?
"""
    claims = ClaimExtractor().from_markdown(md, "slides.md")
    texts = [c.text for c in claims]
    assert "Our model achieves 85% accuracy" in texts
    assert "the baseline scored 50%" in texts
    assert "Our model reaches a macro F1 of 0.80" in texts  # pronoun resolved
    cats = {c.text: c.category for c in claims}
    assert cats["We used a stratified 80/20 train/test split"] == "methodological"
    assert cats["Validation loss decreased steadily"] == "trend"
    assert cats["The embeddings clearly capture sentiment"] == "qualitative"
    assert all(c.context == "Results" for c in claims)
    assert claims[0].source == "slides.md:2"
    assert not any("Questions" in t for t in texts)


def test_markdown_tables_become_claims():
    md = "| Model | Accuracy |\n|---|---|\n| LogReg | 0.85 |\n"
    claims = ClaimExtractor().from_markdown(md, "t.md")
    assert claims[0].text == "Model: LogReg, Accuracy: 0.85"
    assert claims[0].numbers[0].kinds == ["accuracy"]


def test_structure_headings_and_ranges_are_not_measurements():
    md = "**Step 2: Task 1 — Exploratory Data Analysis**\n\n- Step 3: Task 2 — Preprocessing\n- Classes: Small 1–2, Medium 3–6, Large 7+\n"
    claims = ClaimExtractor().from_markdown(md, "nb")
    assert not any(c.text.startswith("Step") for c in claims)
    nums = {n.raw: n.qualifier for c in claims for n in c.numbers}
    assert nums.get("6") == "range" and nums.get("7") == "gt"
