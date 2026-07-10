"""
RQ semantic-ID diagnostic, Analysis 4 -- Dataset dependence of semantic codes.

A. Within- vs. cross-dataset code similarity (shared-prefix length, Hamming
   distance), plus a dataset x dataset heatmap of mean shared-prefix length.
B. Dataset predictability: logistic regression on the 8-level semantic code
   sequence (ordinal + one-hot features) vs. majority-class and shuffled-code
   baselines.

Both A and B are repeated at the coarser "domain" level (fashion/news/wiki/
common, via MBEIR_DATASET_TO_DOMAIN) -- if codes track domain better than
exact dataset identity that's a materially different finding for the paper.
Also runs a chi-square test of independence (dataset x level-1 code value) as
a significance backstop to the classifier-accuracy story.

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python rq_analysis_dataset_dependence.py --genir_dir <repo>
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data.preprocessing.utils import MBEIR_DATASET_TO_DOMAIN
from rq_analysis_utils import (
    GEN_CODE_DIR_DEFAULT, POOL_TO_DATASET_LABEL, hamming_distance, load_all_pools,
    output_dirs, semantic_codes, shared_prefix_length, split_by_modality,
)

RNG_SEED = 2026
N_PAIRS_PER_DATASET = 500
N_SAMPLES_PER_GROUP = 2000  # for the classifier, capped per dataset/domain


def sample_pairs(codes, rng, n_pairs):
    n = len(codes)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    keep = idx_a != idx_b
    return codes[idx_a[keep]], codes[idx_b[keep]]


def within_vs_cross(pools_sem, rng):
    """pools_sem: {label: semantic_codes array}. Returns rows for table4 and
    a label x label matrix of mean shared-prefix length for the heatmap."""
    labels = list(pools_sem.keys())
    rows = [["setting", "group_a", "group_b", "mean_shared_prefix", "std_shared_prefix", "mean_hamming", "std_hamming", "n_pairs"]]

    heatmap = np.zeros((len(labels), len(labels)))
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j < i:
                heatmap[i, j] = heatmap[j, i]
                continue
            if i == j:
                a, b = sample_pairs(pools_sem[la], rng, N_PAIRS_PER_DATASET)
            else:
                na = min(N_PAIRS_PER_DATASET, len(pools_sem[la]))
                nb = min(N_PAIRS_PER_DATASET, len(pools_sem[lb]))
                n = min(na, nb)
                a = pools_sem[la][rng.choice(len(pools_sem[la]), size=n, replace=False)]
                b = pools_sem[lb][rng.choice(len(pools_sem[lb]), size=n, replace=False)]
            spl = shared_prefix_length(a, b)
            ham = hamming_distance(a, b)
            heatmap[i, j] = spl.mean()
            if i == j:
                rows.append(["within", la, la, round(spl.mean(), 4), round(spl.std(), 4), round(ham.mean(), 4), round(ham.std(), 4), len(a)])

    # aggregate cross-dataset rows (off-diagonal) into one summary line, as requested by the assignment
    off_diag_spl, off_diag_ham, total_n = [], [], 0
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i >= j:
                continue
            na = min(N_PAIRS_PER_DATASET, len(pools_sem[la]), len(pools_sem[lb]))
            a = pools_sem[la][rng.choice(len(pools_sem[la]), size=na, replace=False)]
            b = pools_sem[lb][rng.choice(len(pools_sem[lb]), size=na, replace=False)]
            off_diag_spl.append(shared_prefix_length(a, b))
            off_diag_ham.append(hamming_distance(a, b))
            total_n += na
    all_spl = np.concatenate(off_diag_spl)
    all_ham = np.concatenate(off_diag_ham)
    rows.append(["cross", "ALL", "ALL", round(all_spl.mean(), 4), round(all_spl.std(), 4), round(all_ham.mean(), 4), round(all_ham.std(), 4), total_n])

    return rows, heatmap, labels


def plot_heatmap(heatmap, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean shared-prefix length")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_classification_dataset(pools_sem, rng, n_per_group=N_SAMPLES_PER_GROUP):
    X, y = [], []
    for label, codes in pools_sem.items():
        n = min(n_per_group, len(codes))
        idx = rng.choice(len(codes), size=n, replace=False)
        X.append(codes[idx])
        y.extend([label] * n)
    return np.concatenate(X, axis=0), np.array(y)


def run_classifier(X, y, rng, feature_mode):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG_SEED)

    if feature_mode == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore")
        X_train_f = enc.fit_transform(X_train)
        X_test_f = enc.transform(X_test)
    else:
        X_train_f, X_test_f = X_train, X_test

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_f, y_train)
    pred = clf.predict(X_test_f)

    maj = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    maj_pred = maj.predict(X_test)

    # Stratified (within-class) per-column shuffle: permuting each class's
    # rows separately preserves that class's exact per-level marginal
    # frequency distribution while destroying joint cross-level structure
    # within the class. A *global* shuffle (permuting across all classes at
    # once) would additionally destroy the per-class marginals themselves,
    # making the baseline too weak to isolate "is the signal in the joint
    # combination, not the marginals" -- which is the whole point of this
    # baseline. A class with 1 training row is a no-op under permutation,
    # which is the mathematically correct behavior (nothing to shuffle).
    shuffled_X_train = X_train.copy()
    for label in np.unique(y_train):
        row_mask = y_train == label
        for col in range(shuffled_X_train.shape[1]):
            shuffled_X_train[row_mask, col] = rng.permutation(shuffled_X_train[row_mask, col])
    if feature_mode == "onehot":
        shuffled_X_train_f = enc.fit_transform(shuffled_X_train)
        shuffled_clf = LogisticRegression(max_iter=2000).fit(shuffled_X_train_f, y_train)
        shuffled_pred = shuffled_clf.predict(X_test_f)
    else:
        shuffled_clf = LogisticRegression(max_iter=2000).fit(shuffled_X_train, y_train)
        shuffled_pred = shuffled_clf.predict(X_test)

    results = {
        "Logistic Regression": (y_test, pred),
        "Majority baseline": (y_test, maj_pred),
        "Shuffled-code baseline": (y_test, shuffled_pred),
    }
    return results, y_test, pred


def write_classification_table(results, out_path, group_label):
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    rows = []
    for name, (y_true, y_pred) in results.items():
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        rows.append([group_label, name, round(acc, 4), round(f1, 4)])
        print(f"[{group_label}] {name}: accuracy={acc:.4f} macro_f1={f1:.4f}")
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["group_level", "classifier", "accuracy", "macro_f1"])
        writer.writerows(rows)


def plot_confusion(y_test, pred, labels, title, out_path):
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chi_square_backstop(pools_sem, labels):
    """Contingency table: dataset (rows) x level-1 code value (cols)."""
    all_codes = np.concatenate([pools_sem[l][:, 0] for l in labels])
    all_labels = np.concatenate([[l] * len(pools_sem[l]) for l in labels])
    unique_codes = np.unique(all_codes)
    table = np.zeros((len(labels), len(unique_codes)), dtype=int)
    for i, label in enumerate(labels):
        mask = all_labels == label
        vals, counts = np.unique(all_codes[mask], return_counts=True)
        col_idx = np.searchsorted(unique_codes, vals)
        table[i, col_idx] = counts
    chi2, p, dof, _ = chi2_contingency(table)
    return chi2, p, dof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--gen_code_dir", default=GEN_CODE_DIR_DEFAULT)
    parser.add_argument("--out_root", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis")
    args = parser.parse_args()

    gen_code_dir = os.path.join(args.genir_dir, args.gen_code_dir)
    dirs = output_dirs(args.out_root)
    rng = np.random.default_rng(RNG_SEED)

    pools = load_all_pools(gen_code_dir)
    pools_sem = {POOL_TO_DATASET_LABEL[name]: semantic_codes(codes) for name, (codes, _) in pools.items()}

    # MSCOCO's pool mixes images (123,287) and captions (590,392); every other
    # dataset here is pure-image. Left as-is, MSCOCO's "dataset" signal in the
    # classifier/heatmap below would be confounded with modality (image vs.
    # caption embeddings are near-perfectly separable, see analysis3_pca_scatter.png)
    # rather than reflecting genuine cross-dataset semantic structure. Swap in
    # the image-only subset for the PRIMARY comparison so all 6 classes are
    # apples-to-apples; the modality-leakage diagnostic block further below
    # explicitly quantifies what this swap controls for.
    mscoco_codes_raw, _ = pools["mscoco_task0_test"]
    mscoco_split = split_by_modality(mscoco_codes_raw, np.arange(len(mscoco_codes_raw)))
    mscoco_image_sem = semantic_codes(mscoco_split["image"][0])
    mscoco_caption_sem = semantic_codes(mscoco_split["caption"][0])
    pools_sem["MSCOCO"] = mscoco_image_sem

    dataset_labels = sorted(pools_sem.keys())

    # --- A. within vs cross dataset similarity ---
    rows, heatmap, labels = within_vs_cross(pools_sem, rng)
    table4_path = os.path.join(dirs["tables"], "table4_within_cross.csv")
    with open(table4_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Table4 written to {table4_path}")
    plot_heatmap(heatmap, labels, "Mean shared-prefix length by dataset pair", os.path.join(dirs["figures"], "analysis4_dataset_heatmap.png"))

    # --- B. dataset predictability ---
    table5_path = os.path.join(dirs["tables"], "table5_classification.csv")
    if os.path.exists(table5_path):
        os.remove(table5_path)

    X, y = build_classification_dataset(pools_sem, rng)
    results_ordinal, y_test, pred = run_classifier(X, y, rng, feature_mode="ordinal")
    write_classification_table(results_ordinal, table5_path, "dataset_ordinal")
    plot_confusion(y_test, pred, dataset_labels, "Dataset classification confusion matrix (ordinal codes)",
                   os.path.join(dirs["figures"], "analysis4_confusion_matrix.png"))

    results_onehot, _, _ = run_classifier(X, y, rng, feature_mode="onehot")
    write_classification_table(results_onehot, table5_path, "dataset_onehot")

    # --- Extra: domain-level repeat of A+B ---
    pools_domain = {}
    for label, codes in pools_sem.items():
        domain = MBEIR_DATASET_TO_DOMAIN[label]
        pools_domain.setdefault(domain, []).append(codes)
    pools_domain = {d: np.concatenate(cs, axis=0) for d, cs in pools_domain.items()}
    domain_labels = sorted(pools_domain.keys())

    domain_rows, domain_heatmap, _ = within_vs_cross(pools_domain, rng)
    with open(os.path.join(dirs["tables"], "table4_within_cross_domain.csv"), "w", newline="") as f:
        csv.writer(f).writerows(domain_rows)
    plot_heatmap(domain_heatmap, domain_labels, "Mean shared-prefix length by domain pair",
                 os.path.join(dirs["figures"], "analysis4_domain_heatmap.png"))

    X_dom, y_dom = build_classification_dataset(pools_domain, rng)
    results_dom, y_test_dom, pred_dom = run_classifier(X_dom, y_dom, rng, feature_mode="ordinal")
    write_classification_table(results_dom, table5_path, "domain_ordinal")
    plot_confusion(y_test_dom, pred_dom, domain_labels, "Domain classification confusion matrix (ordinal codes)",
                   os.path.join(dirs["figures"], "analysis4_domain_confusion_matrix.png"))

    results_dom_onehot, _, _ = run_classifier(X_dom, y_dom, rng, feature_mode="onehot")
    write_classification_table(results_dom_onehot, table5_path, "domain_onehot")

    # --- Modality-leakage diagnostic: does splitting MSCOCO by modality reveal
    # the classifier was (or wasn't) partly learning modality rather than
    # dataset identity? Part A re-runs the full classifier with MSCOCO split
    # into two classes; Part B isolates how separable modality alone is
    # within MSCOCO, to contextualize Part A's confusion matrix. ---
    pools_sem_modsplit = dict(pools_sem)
    del pools_sem_modsplit["MSCOCO"]
    pools_sem_modsplit["MSCOCO-image"] = mscoco_image_sem
    pools_sem_modsplit["MSCOCO-caption"] = mscoco_caption_sem
    modsplit_labels = sorted(pools_sem_modsplit.keys())

    X_mod, y_mod = build_classification_dataset(pools_sem_modsplit, rng)
    results_modsplit, y_test_mod, pred_mod = run_classifier(X_mod, y_mod, rng, feature_mode="ordinal")
    write_classification_table(results_modsplit, table5_path, "dataset_ordinal_modsplit")
    plot_confusion(y_test_mod, pred_mod, modsplit_labels, "Dataset classification confusion matrix (MSCOCO split by modality)",
                   os.path.join(dirs["figures"], "analysis4_confusion_matrix_modsplit.png"))

    results_modsplit_onehot, _, _ = run_classifier(X_mod, y_mod, rng, feature_mode="onehot")
    write_classification_table(results_modsplit_onehot, table5_path, "dataset_onehot_modsplit")

    # Part B: MSCOCO-image vs MSCOCO-caption alone -- how separable is modality by itself?
    X_sanity, y_sanity = build_classification_dataset({"image": mscoco_image_sem, "caption": mscoco_caption_sem}, rng)
    results_sanity, _, _ = run_classifier(X_sanity, y_sanity, rng, feature_mode="onehot")
    write_classification_table(results_sanity, table5_path, "mscoco_modality_sanity")

    # --- chi-square backstop ---
    chi2, p, dof = chi_square_backstop(pools_sem, dataset_labels)
    print(f"Chi-square test (dataset x level-1 code): chi2={chi2:.1f} dof={dof} p={p:.3e}")
    with open(os.path.join(dirs["tables"], "table5_classification.csv"), "a", newline="") as f:
        csv.writer(f).writerow(["chi_square_backstop", f"chi2={chi2:.1f},dof={dof},p={p:.3e}", "", ""])

    print(f"Table5 written to {table5_path}")


if __name__ == "__main__":
    main()
