"""Generate examples/sample_notebook.ipynb — a SYNTHETIC notebook with planted problems.

Planted issues (what the auditor should find):
  * TF-IDF + StandardScaler fitted on all rows before train_test_split (data leakage)
  * No majority/random baseline next to accuracy/F1
  * Macro F1 = 0.801 vs a 0.80 target, p = 0.048 vs alpha 0.05 (borderline decisions)
  * Validation-loss curve that is not monotonic, missing y-label
  * Bar chart with truncated y-axis; un-seeded t-SNE plot
  * One cell re-executed out of order
The outputs are hand-written (no kernel is run), so they are illustrative, not real results.
"""
import base64
import json
import struct
import zlib
from pathlib import Path


def tiny_png(width=64, height=48, rgb=(70, 110, 200)) -> str:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src, n, outputs=()):
    return {"cell_type": "code", "execution_count": n, "metadata": {}, "source": src, "outputs": list(outputs)}


def stream(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


def result(text, n):
    return {"output_type": "execute_result", "execution_count": n, "metadata": {}, "data": {"text/plain": text}}


def figure(w=640, h=480):
    return {"output_type": "display_data", "metadata": {}, "data": {"image/png": tiny_png(), "text/plain": f"<Figure size {w}x{h} with 1 Axes>"}}


cells = [
    md("# Sentiment classification of product reviews\nTF-IDF + logistic regression on a review corpus."),
    code(
        "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, f1_score\n"
        "from scipy import stats\nimport matplotlib.pyplot as plt\n%matplotlib inline",
        1,
    ),
    code(
        'df = pd.read_csv("reviews.csv")\nprint(f"Loaded {len(df)} documents")\ndf.shape',
        2,
        [stream("Loaded 420 documents\n"), result("(420, 5)", 2)],
    ),
    code('df["label"].value_counts()', 3, [result("positive    252\nnegative    168\nName: label, dtype: int64", 3)]),
    code(
        'vectorizer = TfidfVectorizer(max_features=5000)\nX_text = vectorizer.fit_transform(df["text"])\n'
        'scaler = StandardScaler(with_mean=False)\nX = scaler.fit_transform(X_text)\ny = df["label"]\nprint("Feature matrix:", X.shape)',
        4,
        [stream("Feature matrix: (420, 5000)\n")],
    ),
    code(
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        'print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])',
        5,
        [stream("Train size: 336 Test size: 84\n")],
    ),
    code(
        "model = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)\n"
        'acc = accuracy_score(y_test, preds)\nf1 = f1_score(y_test, preds, average="macro")\n'
        'print(f"Test accuracy: {acc:.4f}")\nprint(f"Macro F1: {f1:.3f}")\nTARGET_F1 = 0.80\nprint("Meets target:", f1 >= TARGET_F1)',
        12,
        [stream("Test accuracy: 0.8452\nMacro F1: 0.801\nMeets target: True\n")],
    ),
    code(
        "scores_lr = [0.84, 0.83, 0.86, 0.82, 0.85]\nscores_svm = [0.81, 0.82, 0.80, 0.83, 0.79]\n"
        "t, p = stats.ttest_ind(scores_lr, scores_svm)\nprint(f\"t = {t:.3f}, p-value = {p:.3f}\")",
        6,
        [stream("t = 2.312, p-value = 0.048\n")],
    ),
    code(
        "val_loss = [0.69, 0.52, 0.44, 0.47, 0.41, 0.43]\n"
        'plt.plot(range(1, 7), val_loss, label="validation loss")\nplt.xlabel("Epoch")\n'
        'plt.title("Validation loss per epoch")\nplt.legend()\nplt.show()',
        7,
        [figure()],
    ),
    code(
        'models = ["LogReg", "SVM", "NB"]\nscores = [0.845, 0.83, 0.81]\nplt.bar(models, scores)\nplt.ylim(0.8, 0.86)\n'
        'plt.ylabel("Accuracy")\nplt.title("Model comparison")\nplt.show()',
        8,
        [figure()],
    ),
    code(
        "from sklearn.manifold import TSNE\nemb = TSNE(n_components=2, perplexity=30).fit_transform(X.toarray())\n"
        'plt.scatter(emb[:, 0], emb[:, 1], c=(y == "positive"), s=8)\nplt.title("t-SNE of TF-IDF features")\nplt.show()',
        9,
        [figure()],
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    out = Path(__file__).with_name("sample_notebook.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
