"""Experiment 9 - other learning algorithms: classical supervised learners, a self-supervised
representation, and an unsupervised (label-free) robustness analysis.

    python scripts/exp9_classical_unsupervised.py --profile pilot

A. Classifiers, evaluated on exactly the test sets of experiment 2 (clean + 14 corruptions x 5):
     HOG + linear SVM, HOG + random forest, HOG + PCA + k-nearest neighbours  (classical, supervised)
     SimCLR features + kNN, SimCLR features + logistic regression            (representation learned
                                                                               without labels; labels
                                                                               only for the classifier)
B. Unsupervised analysis of representations: k-means (k = 43) on the clean test features of each
   model, then
     NMI clean        agreement of the clusters with the true classes (normalised mutual information)
     stability S3/S5  share of corrupted images assigned to the same cluster as their clean original,
                      mean over the 14 corruptions; measured WITHOUT any label
     NMI S5           agreement of the corrupted images' clusters with the classes at severity 5

Writes to results/exp9_classical_unsupervised/: classical.csv/.txt, classical_long.csv, clusters.csv/.txt,
heatmaps/<model>.png (corruption x severity accuracy of every classical learner)
"""

import time
from functools import partial

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from roadlens.data.loading import load_split, u8_to_device
from roadlens.eval import plots
from roadlens.eval.evaluate import build_sets
from roadlens.eval.metrics import corruption_summary
from roadlens.experiment import Experiment, make_parser
from roadlens.runs import ensure_trained, load_model
from roadlens.ssl import embed
from roadlens.utils import get_logger

log = get_logger()


def hog_features(x: np.ndarray, p: dict) -> np.ndarray:
    """HOG of the grey-level image, [N, D] float32 (uint8 NHWC input)."""
    win, block, stride, cell = (int(p[k]) for k in ("win", "block", "stride", "cell"))
    hog = cv2.HOGDescriptor((win, win), (block, block), (stride, stride), (cell, cell), int(p["bins"]))
    out = []
    for img in np.asarray(x):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if g.shape[0] != win:
            g = cv2.resize(g, (win, win), interpolation=cv2.INTER_LINEAR)
        out.append(hog.compute(g).ravel())
    return np.stack(out).astype(np.float32)


def deep_features(model, x: np.ndarray, device) -> np.ndarray:
    return embed(model, u8_to_device(np.asarray(x), device), amp=device.type == "cuda").cpu().numpy()


def evaluate_classifiers(learners, featurize, sets) -> pd.DataFrame:
    """learners: [(name, kind, fitted classifier, fit seconds)]. Every set is featurised once."""
    rows, t_pred, n_pred = [], {n: 0.0 for n, *_ in learners}, 0
    for es in sets:
        x, y = es.load()
        f = featurize(x)
        n_pred += len(y)
        for name, kind, clf, fit_s in learners:
            t0 = time.perf_counter()
            pred = clf.predict(f)
            t_pred[name] += time.perf_counter() - t0
            rows.append({"dataset": es.dataset, "split": es.split, "corruption": es.corruption, "severity": es.severity,
                         "n": len(y), "acc": float((pred == y).mean()), "macro_f1": f1_score(y, pred, average="macro"),
                         "ece": np.nan, "model": name, "kind": kind, "fit_s": fit_s})
    df = pd.DataFrame(rows)
    return df.assign(predict_ms_per_image=df.model.map({n: 1000 * t / n_pred for n, t in t_pred.items()}))


def part_a(exp, sets, norm):
    cfg, c = exp.cfg, exp.cfg["classical"]
    seed = exp.seeds[0]
    xtr, ytr = load_split(cfg, "train")
    t0 = time.perf_counter()
    htr = hog_features(xtr, c["hog"])
    log.info(f"HOG: {htr.shape[1]} features, {len(htr)} training images in {time.perf_counter() - t0:.0f}s")
    learners = [
        ("HOG + linear SVM", "classical supervised",
         make_pipeline(StandardScaler(), LinearSVC(C=float(c["svm_c"]), dual="auto", max_iter=5000, random_state=seed))),
        ("HOG + random forest", "classical supervised",
         RandomForestClassifier(int(c["rf_trees"]), n_jobs=-1, random_state=seed)),
        ("HOG + PCA + kNN", "classical supervised",
         make_pipeline(StandardScaler(), PCA(int(c["pca_dim"]), random_state=seed), KNeighborsClassifier(int(c["knn_k"])))),
    ]
    fitted = []
    for name, kind, clf in learners:
        t0 = time.perf_counter()
        clf.fit(htr, ytr)
        fitted.append((name, kind, clf, time.perf_counter() - t0))
        log.info(f"{name}: fitted in {fitted[-1][3]:.0f}s")
    del htr
    frames = [evaluate_classifiers(fitted, lambda x: hog_features(x, c["hog"]), sets)]

    ssl_run = cfg.get("ssl_run")
    if ssl_run:
        ensure_trained(ssl_run, seed, exp.profile, exp.device)
        model, _ = load_model(ssl_run, seed, exp.device, exp.profile)
        bank = int(cfg.get("knn_bank", 0) or 0)
        idx = np.sort(np.random.RandomState(seed).permutation(len(xtr))[:bank]) if bank and bank < len(xtr) else slice(None)
        ftr = deep_features(model, xtr[idx], exp.device)
        fitted = []
        for name, clf in (("SimCLR features + kNN", KNeighborsClassifier(20, metric="cosine", weights="distance")),
                          ("SimCLR features + logistic regression", LogisticRegression(C=float(c["logreg_c"]), max_iter=2000))):
            t0 = time.perf_counter()
            clf.fit(ftr, ytr[idx])
            fitted.append((name, "self-supervised features (no labels in pre-training)", clf, time.perf_counter() - t0))
        frames.append(evaluate_classifiers(fitted, lambda x: deep_features(model, x, exp.device), sets))
        del model
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(exp.out / "classical_long.csv", index=False, float_format="%.4f")
    plots.plot_corruption_heatmaps(long, "model", exp.out / "heatmaps")
    rows = []
    for name, g in long.groupby("model", sort=False):
        s = corruption_summary(g, norm)
        corr = g[g.corruption != "clean"]
        rows.append({"model": name, "kind": g.kind.iloc[0], "clean_acc": s["clean_acc"], "corr_acc": s["corr_acc"],
                     "s5_acc": 100 * corr[corr.severity == 5].acc.mean(), "mce": s.get("mce"),
                     "corr_f1": s["corr_f1"], "fit_s": g.fit_s.iloc[0],
                     "predict_ms_per_image": g.predict_ms_per_image.iloc[0]})
    table = pd.DataFrame(rows)
    table.to_csv(exp.out / "classical.csv", index=False, float_format="%.3f")
    (exp.out / "classical.txt").write_text(table.round(2).to_string(index=False) + "\n", encoding="utf-8")
    print("\n" + table.round(2).to_string(index=False))


def cluster_scores(f_clean, y, corrupted, k, n_init, seed):
    km = KMeans(k, n_init=n_init, random_state=seed).fit(f_clean)
    base = km.labels_
    out = {"nmi_clean": normalized_mutual_info_score(y, base)}
    for sev, feats in corrupted.items():
        assign = [km.predict(f) for f in feats]
        out[f"stability_s{sev}"] = 100 * float(np.mean([(a == base).mean() for a in assign]))
        out[f"nmi_s{sev}"] = float(np.mean([normalized_mutual_info_score(y, a) for a in assign]))
    return out


def part_b(exp, sets):
    cl, seed = exp.cfg["cluster"], exp.seeds[0]
    clean = next(es for es in sets if es.corruption == "clean")
    x0, y0 = clean.load()
    by_sev = {s: [es for es in sets if es.severity == s] for s in cl["severities"]}
    for group in by_sev.values():
        for es in group:
            assert (es.load()[1] == y0).all(), "corrupted sets must be aligned with the clean test set"
    rows = []
    hogp = exp.cfg["classical"]["hog"]

    def hog_pca(x, pca=None):
        f = hog_features(x, hogp)
        return f if pca is None else pca.transform(f)

    pca = make_pipeline(StandardScaler(), PCA(int(exp.cfg["classical"]["pca_dim"]), random_state=seed)).fit(hog_features(x0, hogp))
    rows.append({"representation": "HOG (+ PCA)", "kind": "hand-crafted",
                 **cluster_scores(hog_pca(x0, pca), y0, {s: [hog_pca(es.load()[0], pca) for es in g] for s, g in by_sev.items()},
                                  int(cl["k"]), int(cl["n_init"]), seed)})
    for run in cl["runs"]:
        try:
            ensure_trained(run, seed, exp.profile, exp.device)
        except Exception as e:                       # a run that cannot be trained here is reported, not fatal
            log.warning(f"skipping {run}: {e}")
            continue
        model, cfg = load_model(run, seed, exp.device, exp.profile)
        feat = partial(deep_features, model, device=exp.device)
        kind = "self-supervised" if cfg["train"].get("objective") == "simclr" else "supervised network"
        rows.append({"representation": exp.label(run), "kind": kind,
                     **cluster_scores(feat(x0), y0, {s: [feat(es.load()[0]) for es in g] for s, g in by_sev.items()},
                                      int(cl["k"]), int(cl["n_init"]), seed)})
        del model
        if exp.device.type == "cuda":
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(exp.out / "clusters.csv", index=False, float_format="%.4f")
    (exp.out / "clusters.txt").write_text(table.round(3).to_string(index=False) + "\n", encoding="utf-8")
    print("\n" + table.round(3).to_string(index=False))


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp9_classical_unsupervised.yaml").parse_args(),
                     "exp9_classical_unsupervised")
    exp.cfg.setdefault("labels", {})["resnet18_simclr"] = "SimCLR backbone (no labels)"
    exp.cfg["labels"].update({"resnet18_v3": "ResNet-18 v3", "resnet18_v3_ssl": "ResNet-18 v3, SimCLR pre-trained",
                              "convnext_atto_v3": "ConvNeXt-Atto v3", "student_v3": "v3 student (ResNet-18)"})
    sets = build_sets(exp.cfg, "gtsrb_c")
    norm = exp.normalizer_table("gtsrb_c")
    part_a(exp, sets, norm)
    part_b(exp, sets)


if __name__ == "__main__":
    main()
