# Sentiment Analysis of Product Reviews

---

## Data

- We collected 3,020 documents from an online review site.
- The dataset is imbalanced: 252 reviews are positive and 168 are negative.
- We used a stratified 80/20 train/test split.
- Features were standardized after splitting to avoid leakage.

---

## Results

- Our model achieves 85% accuracy on the held-out test set.
- It reaches a macro F1 of 0.80, which exceeds our 0.80 target.
- The improvement over SVM is statistically significant (p = 0.048).
- Validation loss decreased steadily across all 6 epochs.

---

## Analysis

- The t-SNE plot demonstrates that the model clearly separates positive and negative reviews.
- Logistic regression outperforms the random baseline by 35 points.

---

## Questions?
