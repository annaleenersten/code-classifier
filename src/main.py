import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit


SEED = 42
MAX_CHARS = 20_000      # cap very long code samples so TF-IDF stays fast
RESULTS_DIR = "results"
PROCESSED_DIR = "data/processed"   # written by data_prep.py

# Class names shown on the confusion matrix, as [label 0, label 1].
CLASS_NAMES = {"vuln": ["non-vulnerable", "vulnerable"],
               "bugs": ["correct", "buggy"]}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def read_processed(name):
    """Read one cleaned CSV, with a helpful message if data_prep.py wasn't run."""
    path = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run `python data_prep.py` first to "
                         f"create the cleaned files.")
    return pd.read_csv(path)


def get_data(task):
    """Return (train, validation, test) DataFrames with 'code' and 'label' columns."""
    if task == "vuln":
        train = read_processed("vuln_train.csv")
        test = read_processed("vuln_test.csv")
        # The vulnerability data has no validation file, so carve one out of
        # train. Splitting by repo keeps related functions on the same side.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tr_idx, va_idx = next(gss.split(train, groups=train["repo"]))
        train, val = train.iloc[tr_idx], train.iloc[va_idx]
    else:
        # The bug data was already split by repository in data_prep.py.
        train = read_processed("bugs_train.csv")
        val = read_processed("bugs_val.csv")
        test = read_processed("bugs_test.csv")

    train, val, test = (d.reset_index(drop=True).copy() for d in (train, val, test))
    for d in (train, val, test):
        d["code"] = d["code"].astype(str).str.slice(0, MAX_CHARS)
    return train, val, test


# ----------------------------------------------------------------------
# Modeling and metrics
# ----------------------------------------------------------------------
def score(name, y_true, y_pred, y_prob):
    """Compute every metric we report for one model."""
    return {"model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),            # priority metric
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob)}


def pick_threshold(y_val, prob_val):
    """Choose the probability cutoff that maximizes F2 on the validation set.

    F2 weights recall twice as much as precision, matching our goal of
    missing as few bugs/vulnerabilities as possible.
    """
    grid = np.arange(0.10, 0.91, 0.05)
    return max(grid, key=lambda t: fbeta_score(y_val, (prob_val >= t).astype(int),
                                               beta=2, zero_division=0))


def train_and_evaluate(train, val, test, n_trees, max_features):
    # Turn code text into numbers: TF-IDF over code tokens (words and symbols,
    # one- and two-token sequences). Fit on TRAIN only, then apply to val/test.
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\S",
                            lowercase=False, ngram_range=(1, 2),
                            max_features=max_features, min_df=3, sublinear_tf=True)
    X_tr = tfidf.fit_transform(train["code"])
    X_va = tfidf.transform(val["code"])
    X_te = tfidf.transform(test["code"])
    y_tr, y_va, y_te = train["label"], val["label"], test["label"]

    rows, probs = [], {}

    # 1. Baseline: always predicts the most common class.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    rows.append(score("Baseline (majority)", y_te, dummy.predict(X_te),
                      dummy.predict_proba(X_te)[:, 1]))

    # 2. Logistic regression: a simple linear model to compare against.
    print("Training logistic regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
    probs["Logistic Regression"] = lr.predict_proba(X_te)[:, 1]
    rows.append(score("Logistic Regression", y_te, lr.predict(X_te),
                      probs["Logistic Regression"]))

    # 3. Random Forest. class_weight="balanced" makes mistakes on the rarer
    #    class cost more, which helps recall when classes are uneven.
    print(f"Training Random Forest ({n_trees} trees)...")
    rf = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced",
                                n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
    probs["Random Forest"] = rf.predict_proba(X_te)[:, 1]
    rows.append(score("Random Forest (0.50)", y_te, rf.predict(X_te),
                      probs["Random Forest"]))

    # 4. Same Random Forest, but with a threshold tuned for recall on validation.
    p_val = rf.predict_proba(X_va)[:, 1]
    thr = pick_threshold(y_va, p_val)

    # Guard: if the chosen threshold flags almost everything, the model has no
    # real signal and "tuning" just means "predict positive for all". Fall back to 0.50.
    flagged = (p_val >= thr).mean()
    if flagged > 0.95:
        print(f"WARNING: threshold {thr:.2f} flags {flagged:.0%} of validation "
              f"samples, so the model cannot separate the classes. Using 0.50 instead.")
        thr = 0.50

    tuned_pred = (probs["Random Forest"] >= thr).astype(int)
    tuned_name = f"Random Forest (tuned {thr:.2f})"
    rows.append(score(tuned_name, y_te, tuned_pred, probs["Random Forest"]))
    print(f"Threshold used: {thr:.2f}")

    return pd.DataFrame(rows), y_te, probs, tuned_pred, thr


# ----------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------
def plot_results(task, metrics, y_te, probs, tuned_pred, thr, path, show=False):
    """One figure with three panels: metric bars, confusion matrix, PR curve."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(f"{'Vulnerability' if task == 'vuln' else 'Bug'} classifier: "
                 f"test-set evaluation", fontsize=14)

    # Panel 1: recall / precision / F1 for every model, side by side.
    ax = axes[0]
    names = list(metrics["model"])
    measures = ["recall", "precision", "f1"]
    width = 0.8 / len(measures)
    x = np.arange(len(names))
    for i, m in enumerate(measures):
        bars = ax.bar(x + i * width, metrics[m], width, label=m.capitalize())
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("Recall, precision and F1 by model")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: confusion matrix for the recall-tuned Random Forest.
    ax = axes[1]
    cm = confusion_matrix(y_te, tuned_pred)
    ax.imshow(cm, cmap="Blues")
    classes = CLASS_NAMES[task]
    ax.set_xticks([0, 1], [f"pred\n{c}" for c in classes])
    ax.set_yticks([0, 1], [f"actual\n{c}" for c in classes])
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix: Random Forest (tuned {thr:.2f})\n"
                 f"bottom-left = missed (false negatives)", fontsize=10)

    # Panel 3: precision-recall curves, with the tuned operating point marked.
    ax = axes[2]
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_te, p)
        ax.plot(rec, prec, label=name)
    row = metrics[metrics["model"].str.startswith("Random Forest (tuned")].iloc[0]
    ax.scatter([row["recall"]], [row["precision"]], color="red", zorder=5,
               label=f"RF at threshold {thr:.2f}")
    ax.axhline(float(np.mean(y_te)), ls="--", color="gray", lw=1,
               label="No-skill baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved chart to {path}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["vuln", "bugs"], required=True)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    train, val, test = get_data(args.task)
    metrics, y_te, probs, tuned_pred, thr = train_and_evaluate(
        train, val, test, args.trees, args.max_features)

    # Print and save the full metrics table.
    print("\nTest-set results:")
    print(metrics.round(3).to_string(index=False))
    csv_path = os.path.join(RESULTS_DIR, f"{args.task}_metrics.csv")
    metrics.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    plot_results(args.task, metrics, y_te, probs, tuned_pred, thr,
                 os.path.join(RESULTS_DIR, f"{args.task}_evaluation.png"),
                 show=args.show)


if __name__ == "__main__":
    main()