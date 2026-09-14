"""
Consistency test for the ECIR short paper (2026-09-14 revision): every number printed in
paper-ecir/sections/*.tex that comes from our own experiments is recomputed here from the
raw sources (result TSVs, CSVs, logs, qrels, candidate pools, configs) and must appear
verbatim in the LaTeX. Exits with status 1 if any check fails.

External numbers (GENIUS's published 41.2% COCO-only R@1, arXiv v2 Table 12) are not
checked here.

Run with: /home/tcetoje/miniconda3/envs/genius2/bin/python check_shortpaper_numbers.py
"""
import csv
import os
import re
import statistics
import sys

from scipy import stats
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_shortpaper_table1 as t1  # noqa: E402
import paired_query_mcnemar_coco as mc  # noqa: E402

REPO = "/home/tcetoje/GENIUS-CVPR25"
SECTIONS = f"{REPO}/paper-ecir/sections"
FS = "/fnwi_fs/ivi/irlab/personal/tcetoje"
MBEIR = f"{FS}/mbeir_data"
LOGS = "/home/tcetoje/logs"
EVAL_CFG = f"{REPO}/src/models/generative_retriever/configs_scripts/large/eval/inbatch"
S2_CFG = f"{REPO}/src/models/generative_retriever/configs_scripts/large/train/inbatch"
S1_CFG = f"{REPO}/src/models/residual_quantization/configs_scripts/large/train/inbatch"

ABS, SETUP, F1, F2, F3, LIM = ("00_abstract.tex", "03_experimental_setup.tex",
                               "04_finding1_balance_tradeoff.tex", "05_finding2_direction_dependence.tex",
                               "06_finding3_tokenizer_transfer.tex", "07_limitations_conclusion.tex")
TEX = {n: open(os.path.join(SECTIONS, n)).read() for n in os.listdir(SECTIONS) if n.endswith(".tex")}
failures = []


def check(label, token, file):
    ok = token in TEX[file]
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {token!r} in {file}")
    if not ok:
        failures.append(label)


def check_true(label, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        failures.append(label)


def mean(v):
    return statistics.mean(v)


def rel(a, b):
    return 100 * (a / b - 1)


def ms(v):
    return f"{mean(v):.2f}\\pm{statistics.stdev(v):.2f}"


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def strip_fmt(cell):
    cell = re.sub(r"\\(?:mathbf|textbf)\{(.*)\}", r"\1", cell.strip())
    return cell.replace("$", "").strip()


def latex_rows(text):
    rows = []
    for line in text.splitlines():
        if "&" in line and line.rstrip().endswith("\\\\") and "toprule" not in line:
            rows.append([strip_fmt(c) for c in line.rstrip()[:-2].split("&")])
    return rows[1:]  # drop header row


def ckpt_epoch(path, key):
    with open(path) as f:
        return int(re.search(key + r"\D*?(\d+)", f.read()).group(1))


def main():
    # ---------------- Table 1 ----------------
    T2I, I2T, IT2I = t1.T2I, t1.I2T, t1.IT2I
    runs = {("MSCOCO", v): t1.coco_runs(v) for v in ("vanilla", "weak", "medium", "strong")}
    runs.update({("VisualNews", v): t1.visualnews_runs(v) for v in ("vanilla", "strong")})
    runs.update({("FashionIQ", v): t1.fashioniq_runs(v) for v in ("vanilla", "weak", "medium", "strong")})
    id_dir = {"MSCOCO": "rq_analysis_coco_variants", "VisualNews": "rq_analysis_visualnews_variants",
              "FashionIQ": "rq_analysis_fashioniq_variants"}
    ent, coll = {}, {}
    rows, dataset = latex_rows(TEX[F1]), None
    for cells, key in zip(rows, runs):
        dataset = cells[0] or dataset
        variant = cells[1].split()[0]
        check_true(f"Table 1 row order {key}", (dataset, variant) == key)
        prim = IT2I if key[0] == "FashionIQ" else T2I
        kmetric = "Recall@10" if key[0] == "FashionIQ" else "Recall@5"
        r = runs[key]
        ent[key], coll[key] = t1.id_structure(id_dir[key[0]], key[1])
        expected = [f"{ent[key]:.2f}", f"{coll[key]:.1f}", ms([x[(prim, 'Recall@1')] for x in r]),
                    ms([x[(prim, kmetric)] for x in r]),
                    "--" if key[0] == "FashionIQ" else ms([x[(I2T, 'Recall@1')] for x in r])]
        check_true(f"Table 1 {key}: {expected} == {cells[2:]}", cells[2:] == expected)
        check_true(f"Table 1 {key}: n seeds", len(r) == (5 if key[0] == "MSCOCO" and key[1] in ("vanilla", "strong") else 3))

    a1 = {k: [x[(IT2I if k[0] == "FashionIQ" else T2I, "Recall@1")] for x in v] for k, v in runs.items()}
    for d in ("MSCOCO", "VisualNews", "FashionIQ"):
        van = ent[(d, "vanilla")]
        check_true(f"{d}: every regularized H < vanilla H", all(e < van for k, e in ent.items() if k[0] == d and k[1] != "vanilla"))
    cc = [coll[("MSCOCO", v)] for v in ("vanilla", "weak", "medium", "strong")]
    check_true("MSCOCO collision increases with lambda", cc == sorted(cc))
    check("collision range",f"from ${cc[0]:.1f}\\%$ to ${cc[-1]:.1f}\\%$", F1)

    cv, cs = ("MSCOCO", "vanilla"), ("MSCOCO", "strong")
    r5 = {k: mean([x[(T2I, "Recall@5")] for x in runs[k]]) for k in (cv, cs)}
    check("MSCOCO strong vs vanilla R@1", f"(${mean(a1[cs]):.2f}\\%$ vs.\\ ${mean(a1[cv]):.2f}\\%$, ${rel(mean(a1[cs]), mean(a1[cv])):+.1f}\\%$; R@5 ${rel(r5[cs], r5[cv]):+.1f}\\%$)", F1)
    check("weak/medium drops", f"by ${-rel(mean(a1[('MSCOCO', 'weak')]), mean(a1[cv])):.0f}\\%$ and ${-rel(mean(a1[('MSCOCO', 'medium')]), mean(a1[cv])):.0f}\\%$", F1)
    check("abstract: up to medium drop", f"up to ${-rel(mean(a1[('MSCOCO', 'medium')]), mean(a1[cv])):.0f}\\%$", ABS)
    check("decoder-seed std bound", f"(std $\\leq{max(statistics.stdev(a1[cv]), statistics.stdev(a1[cs])):.2f}$)", F1)
    t = stats.ttest_ind(a1[cs], a1[cv], equal_var=False)
    check_true(f"decoder-seed Welch p tiny ({t.pvalue:.1e}) -- not printed, sanity only", t.pvalue < 1e-6)

    # tokenizer replication
    tok = {v: [t1.read_tsv(f"{t1.RESULTS}/coco_seed2023_reeval/{v}_seed2023_reeval.tsv")[(T2I, "Recall@1")]]
           + [t1.read_tsv(f"{t1.RESULTS}/coco_stage1_seedvar_and_img0txt3/{v}_s1seed{s}.tsv")[(T2I, "Recall@1")] for s in (7, 13)]
           for v in ("vanilla", "strong")}
    t = stats.ttest_ind(tok["strong"], tok["vanilla"], equal_var=False)
    trel = rel(mean(tok["strong"]), mean(tok["vanilla"]))
    check("tokenizer means + Welch", f"Strong reaches ${ms(tok['strong'])}\\%$ and vanilla ${ms(tok['vanilla'])}\\%$ (${trel:+.1f}\\%$, Welch's $t$-test $p={t.pvalue:.3f}$)", F1)
    check_true("every strong tokenizer beats every vanilla tokenizer", min(tok["strong"]) > max(tok["vanilla"]))
    check("abstract tokenizer gain", f"${trel:+.0f}\\%$ relative across three independently trained tokenizers", ABS)

    # McNemar on the kept run files
    qrel = mc.load_qrels(mc.QRELS)
    rk = {v: mc.load_rank1(mc.RUN.format(d)) for v, d in (("vanilla", "CocoOnlyVanilla"), ("strong", "CocoOnlyStrong"))}
    hv = [rk["vanilla"].get(q) in qrel[q] for q in qrel]
    hs = [rk["strong"].get(q) in qrel[q] for q in qrel]
    b = sum(x and not y for x, y in zip(hv, hs))
    c = sum(y and not x for x, y in zip(hv, hs))
    p = binomtest(b, b + c, 0.5).pvalue
    check("McNemar counts", f"({c:,} queries are retrieved at rank 1 only by strong and {b:,} only by vanilla; McNemar's exact test, $p<10^{{-12}}$)", F1)
    check_true(f"McNemar p={p:.2e} < 1e-12", p < 1e-12)

    with open(f"{HERE}/table_tie_break_sensitivity.csv") as f:
        tb = list(csv.DictReader(f))
    maxdiff = max(abs(float(r["current_R@1_pct"]) - float(r[k])) for r in tb
                  for k in ("expected_random_tiebreak_R@1_pct", "adversarial_worst_case_R@1_pct"))
    check("tie-break bound", f"changes R@1 by at most ${round(maxdiff + 1e-9, 2):.2f}$ points", F1)

    vv, vs = ("VisualNews", "vanilla"), ("VisualNews", "strong")
    vrel = rel(mean(a1[vs]), mean(a1[vv]))
    check("VisualNews T->I", f"from ${mean(a1[vv]):.2f}\\%$ to ${mean(a1[vs]):.2f}\\%$ (${vrel:.0f}\\%$, decoder seeds only)", F1)
    check("abstract VisualNews drop", f"reduces it by ${-vrel:.0f}\\%$ on VisualNews", ABS)
    fr10 = {v: mean([x[(IT2I, "Recall@10")] for x in runs[("FashionIQ", v)]]) for v in ("vanilla", "weak", "medium", "strong")}
    reg = [fr10[v] for v in ("weak", "medium", "strong")]
    check("FashionIQ R@10 range", f"R@10 from ${fr10['vanilla']:.2f}\\%$ to ${min(reg):.2f}$--${max(reg):.2f}\\%$", F1)

    # ---------------- Setup ----------------
    n = {k: count_lines(p) for k, p in {
        "coco_t2i_q": f"{MBEIR}/query/test/mbeir_mscoco_task0_test.jsonl",
        "coco_img": f"{MBEIR}/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl",
        "coco_i2t_q": f"{MBEIR}/query/test/mbeir_mscoco_task3_test.jsonl",
        "coco_txt": f"{MBEIR}/cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl",
        "vn_t2i_q": f"{MBEIR}/query/test/mbeir_visualnews_task0_test.jsonl",
        "vn_img": f"{MBEIR}/cand_pool/local/mbeir_visualnews_task0_test_cand_pool.jsonl",
        "vn_i2t_q": f"{MBEIR}/query/test/mbeir_visualnews_task3_test.jsonl",
        "vn_txt": f"{MBEIR}/cand_pool/local/mbeir_visualnews_task3_test_cand_pool.jsonl",
        "fiq_q": f"{MBEIR}/query/test/mbeir_fashioniq_task7_test.jsonl",
        "fiq_img": f"{MBEIR}/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl"}.items()}
    check("MSCOCO pools", f"MSCOCO has {n['coco_t2i_q']:,} T$\\to$I queries over {n['coco_img']:,} images and {n['coco_i2t_q']:,} I$\\to$T queries over {n['coco_txt']:,} captions", SETUP)
    check("VisualNews pools", f"VisualNews has {n['vn_t2i_q']:,} T$\\to$I queries over {n['vn_img']:,} images and {n['vn_i2t_q']:,} I$\\to$T queries over {n['vn_txt']:,} captions", SETUP)
    check("FashionIQ pool", f"FashionIQ has {n['fiq_q']:,} queries over {n['fiq_img']:,} images", SETUP)

    ep = {d: (ckpt_epoch(f"{S1_CFG}/inbatch_{d}_vanilla.yaml", "num_train_epochs"),
              ckpt_epoch(f"{S2_CFG}/inbatch_{d2}_vanilla.yaml", "rq_clip_large_epoch_"),
              ckpt_epoch(f"{S2_CFG}/inbatch_{d2}_vanilla.yaml", "num_train_epochs"))
          for d, d2 in (("coco", "coco_only"), ("visualnews", "visualnews_only"), ("fashioniq", "fashioniq_only"))}
    check_true(f"VisualNews epochs equal MSCOCO epochs {ep}", ep["coco"] == ep["visualnews"])
    check("tokenizer epochs", f"Tokenizers are trained for {ep['coco'][0]} epochs (FashionIQ: {ep['fashioniq'][0]}; we use epoch {ep['coco'][1]}/{ep['fashioniq'][1]})", SETUP)
    check("decoder epochs", f"for {ep['coco'][2]} epochs (FashionIQ: {ep['fashioniq'][2]})", SETUP)
    eval_eps = set()
    for v in ("vanilla", "weak", "medium", "strong"):
        for s in ("", "_seed7", "_seed13") + (("_seed21", "_seed42") if v in ("vanilla", "strong") else ()):
            eval_eps.add(ckpt_epoch(f"{EVAL_CFG}/config_eval_coco_only_{v}{s}.yaml", "genius_t5small_epoch_"))
    for v in ("vanilla", "strong"):
        for s in ("", "_seed7", "_seed13"):
            eval_eps.add(ckpt_epoch(f"{EVAL_CFG}/config_eval_visualnews_only_{v}{s}.yaml", "genius_t5small_epoch_"))
    check_true(f"MSCOCO/VisualNews eval epochs all equal {eval_eps}", len(eval_eps) == 1)
    check("MSCOCO/VisualNews eval epoch", f"evaluated at epoch {eval_eps.pop()}", SETUP)
    fiq_eps = {v: {ckpt_epoch(f"{EVAL_CFG}/config_eval_fashioniq_only_{v}_{s}.yaml", "genius_t5small_epoch_")
                   for s in (("seed7", "seed13") + (("seedrepaircheck",) if v in ("weak", "medium") else ()))}
               for v in ("vanilla", "weak", "medium", "strong")}
    with open(f"{FS}/rq_analysis_fashioniq_variants/tables/table2_retrieval_performance.csv") as f:
        best = {r["method"]: int(r["best_epoch"]) for r in csv.DictReader(f)}
    check_true(f"FashionIQ eval epochs {fiq_eps} match seed-2023 best epochs {best} (vanilla/strong)",
               all(fiq_eps[v] == {best[v]} for v in ("vanilla", "strong")) and fiq_eps["weak"] == fiq_eps["medium"] == {10})
    check("FashionIQ epochs text", f"({best['vanilla']} for vanilla, {best['weak']} otherwise)", SETUP)
    sweep = {v: [float(r["IT->I Recall@10"]) for r in csv.DictReader(open(f"{REPO}/retrieval_results/fashioniq_epoch_sweep_{v}/fashioniq_recall_by_epoch.csv"))]
             for v in ("vanilla", "weak", "medium", "strong")}
    check("FashionIQ sweep bound", f"R@10 ${min(sweep['vanilla']):.2f}\\%$ vs.\\ at most ${max(max(sweep[v]) for v in ('weak', 'medium', 'strong')):.2f}\\%$", SETUP)
    check("absolute: vanilla R@1", f"reaches ${mean(a1[cv]):.1f}\\%$ T$\\to$I R@1", SETUP)
    dense = float(re.search(r"\[T2I_task0\].*?R@1=([\d.]+)%", open(f"{LOGS}/dense_clip_sf_raw_baseline_346557.log").read()).group(1))
    check("dense CLIP-SF baseline", f"on our pool reaches ${dense:.1f}\\%$", SETUP)

    # ---------------- Finding 2 ----------------
    b1 = {k: [x[(I2T, "Recall@1")] for x in v] for k, v in runs.items() if k[0] != "FashionIQ"}
    regb = [mean(b1[("MSCOCO", v)]) for v in ("weak", "medium", "strong")]
    check("MSCOCO I->T", f"vanilla reaches ${mean(b1[cv]):.2f}\\%$ I$\\to$T R@1", F2)
    check("MSCOCO regularized I->T range", f"drops to ${min(regb):.2f}$--${max(regb):.2f}\\%$", F2)

    def mean_pos(path):
        counts = {}
        for line in open(path):
            q = line.split()[0]
            counts[q] = counts.get(q, 0) + 1
        return mean(counts.values())
    coco_chance = 100 * mean_pos(f"{MBEIR}/qrels/test/mbeir_mscoco_task3_test_qrels.txt") / n["coco_txt"]
    vn_chance = 100 * mean_pos(f"{MBEIR}/qrels/test/mbeir_visualnews_task3_test_qrels.txt") / n["vn_txt"]
    words = {4: "four", 5: "five", 6: "six"}
    coco_pos = round(mean_pos(f"{MBEIR}/qrels/test/mbeir_mscoco_task3_test_qrels.txt"))
    check("MSCOCO chance", f"with {n['coco_txt']:,} captions and about {words.get(coco_pos, coco_pos)} relevant", F2)
    check("MSCOCO chance value", f"random ranking gives about ${coco_chance:.2f}\\%$", F2)
    check("VisualNews I->T + chance", f"from ${mean(b1[vv]):.2f}\\%$ to ${mean(b1[vs]):.2f}\\%$ (chance: ${vn_chance:.3f}\\%$)", F2)
    check("weak T->I cost", f"costs MSCOCO T$\\to$I ${-rel(mean(a1[('MSCOCO', 'weak')]), mean(a1[cv])):.0f}\\%$", F2)
    with open(f"{HERE}/table_direction_aware_seed_variance.csv") as f:
        da = {r["variant"]: r for r in csv.DictReader(f)}
    i0t3 = [t1.read_tsv(f"{t1.RESULTS}/coco_stage1_seedvar_and_img0txt3/img0txt3{s}.tsv") for s in ("", "_seed7", "_seed13")]
    i2t_da = [f"{float(da[v]['i2t_r1_mean']):.2f}\\pm{float(da[v]['i2t_r1_std']):.2f}" for v in ("img3txt0", "img3txt0p3")] + [ms([x[(I2T, 'Recall@1')] for x in i0t3])]
    t2i_da = [f"{float(da[v]['t2i_r1_mean']):.2f}\\pm{float(da[v]['t2i_r1_std']):.2f}" for v in ("img3txt0", "img3txt0p3")] + [ms([x[(T2I, 'Recall@1')] for x in i0t3])]
    check("per-modality I->T", "(" + ", ".join(f"${x}\\%$" for x in i2t_da) + ")", F2)
    check("per-modality T->I", "(" + ", ".join(f"${x}\\%$" for x in t2i_da) + f" vs.\\ ${mean(a1[cv]):.2f}\\%$)", F2)

    # ---------------- Finding 3 ----------------
    def probe(path):
        out = {}
        for m in re.finditer(r"\| (\w+) \| (T2I|I2T)_task\d \| ([\d.]+)% \|", open(path).read()):
            out.setdefault((m.group(1), m.group(2)), float(m.group(3)))
        return out
    pc = probe(f"{LOGS}/genius_probe_collapse_localize_335034.log")
    for cells, v in zip(latex_rows(TEX[F3]), ("vanilla", "weak", "medium", "strong")):
        expected = [v, f"{pc[(v, 'T2I')]:.2f}", f"{pc[(v, 'I2T')]:.2f}",
                    f"{mean(a1[('MSCOCO', v)]):.2f}", f"{mean(b1[('MSCOCO', v)]):.2f}"]
        check_true(f"Table 2 {v}: {expected} == {cells}", cells == expected)
    reg_probe_i2t = max(pc[(v, "I2T")] for v in ("weak", "medium", "strong"))
    check("probe regularized I->T bound", f"($\\leq{reg_probe_i2t:.2f}\\%$), while vanilla keeps ${pc[('vanilla', 'I2T')]:.2f}\\%$", F3)
    check("probe strong vs vanilla", f"strong (${pc[('strong', 'T2I')]:.2f}\\%$) is below vanilla (${pc[('vanilla', 'T2I')]:.2f}\\%$)", F3)
    check_true("probe: strong T->I below vanilla", pc[("strong", "T2I")] < pc[("vanilla", "T2I")])
    check("probe-to-pipeline cost", f"costs vanilla ${-rel(mean(a1[cv]), pc[('vanilla', 'T2I')]):.0f}\\%$ of its R@1 but strong only ${-rel(mean(a1[cs]), pc[('strong', 'T2I')]):.0f}\\%$", F3)
    pv = probe(f"{LOGS}/genius_probe_collapse_localize_visualnews_335182.log")
    check("VisualNews probe", f"I$\\to$T falls from ${pv[('vanilla', 'I2T')]:.2f}\\%$ to ${pv[('strong', 'I2T')]:.2f}\\%$ and T$\\to$I from ${pv[('vanilla', 'T2I')]:.2f}\\%$ to ${pv[('strong', 'T2I')]:.2f}\\%$ (${rel(pv[('strong', 'T2I')], pv[('vanilla', 'T2I')]):.0f}\\%$, pipeline ${vrel:.0f}\\%$)", F3)
    with open(f"{FS}/rq_analysis_coco_subsampled74k_variants/tables/pool_size_confound_table.csv") as f:
        ps = list(csv.DictReader(f))
    util = {(r["source"], r["variant"]): float(r["utilization_pct_mean"]) for r in ps}
    ratios = [util[("COCO-subsampled74K", v)] / util[("FashionIQ-real", v)] for v in ("vanilla", "weak", "medium", "strong")]
    sub_n = {int(r["n_candidates"]) for r in ps if r["source"] == "COCO-subsampled74K"}
    check_true(f"subsample size {sub_n} == FashionIQ pool {n['fiq_img']}", sub_n == {n["fiq_img"]})
    check("pool-size ratio", f"${min(ratios):.1f}$ to ${max(ratios):.1f}$ times", F3)

    # ---------------- Limitations ----------------
    m = re.search(r"=== coco / vanilla ===.*?image=([\d.]+)\s+text=([\d.]+)", open(f"{LOGS}/genius_anisotropy_test_335048.log").read(), re.S)
    check("anisotropy", f"${float(m.group(2)):.2f}$ vs.\\ ${float(m.group(1)):.2f}$", LIM)

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
