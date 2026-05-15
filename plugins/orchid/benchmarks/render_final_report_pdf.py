import argparse
import csv
import os
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_csv(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _f(x, default=0.0) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _i(x, default=0) -> int:
    try:
        if x is None or x == "":
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _img(path: str):
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def _page_text(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.05
    for ln in lines:
        ax.text(0.03, y, ln, fontsize=11, va="top", wrap=True)
        y -= 0.03
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            y = 0.97
    pdf.savefig(fig)
    plt.close(fig)


def _page_image(pdf: PdfPages, title: str, img_path: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, title, fontsize=16, fontweight="bold", va="top")
    if os.path.exists(img_path):
        ax.imshow(_img(img_path))
        ax.set_position([0.05, 0.05, 0.90, 0.88])
    else:
        ax.text(0.02, 0.90, f"Missing image: {img_path}", fontsize=12, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=benchmark_artifact())
    ap.add_argument("--out", default=benchmark_artifact("final_report.pdf"))
    args = ap.parse_args()

    a = os.path.abspath(str(args.artifacts))
    out_pdf = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    suite_csv = os.path.join(a, "simple_suite_more.csv")
    seed_csv = os.path.join(a, "bs4_p64_s128_seed_sweep.csv")
    prefill_csv = os.path.join(a, "prefill_only_sweep.csv")
    perf_csv = os.path.join(a, "perf_variance.csv")

    suite = _read_csv(suite_csv)
    seed = _read_csv(seed_csv)
    prefill = _read_csv(prefill_csv)
    perf = _read_csv(perf_csv)

    ratios_eager = []
    ratios_graph = []
    ratios_trt_cg_over_graph = []
    trt_cg_over_trt = []
    vllm_graph_over_eager = []
    for r in suite:
        te = _f(r.get("trt_decode_tok_s"))
        tcg = _f(r.get("trt_decode_tok_s_cg"))
        ve = _f(r.get("vllm_decode_tok_s"))
        vg = _f(r.get("vllm_decode_tok_s_graph"))
        if ve > 0:
            ratios_eager.append(te / ve)
        if vg > 0:
            ratios_graph.append(te / vg)
        if vg > 0 and tcg > 0:
            ratios_trt_cg_over_graph.append(tcg / vg)
        if te > 0 and tcg > 0:
            trt_cg_over_trt.append(tcg / te)
        if ve > 0 and vg > 0:
            vllm_graph_over_eager.append(vg / ve)

    seed_tm = [_f(r.get("token_match_rate")) for r in seed]
    seed_tf = [_f(r.get("tf_next_token_match_rate")) for r in seed]
    seed_pmin = [_i(r.get("prefix_match_min")) for r in seed]

    prefill_lens = [_i(r.get("prefill_len")) for r in prefill]
    prefill_acc = [_f(r.get("prefill_next_token_match_rate")) for r in prefill]

    with PdfPages(out_pdf) as pdf:
        avg_e = sum(ratios_eager) / max(1, len(ratios_eager))
        emin = min(ratios_eager) if ratios_eager else 0.0
        emax = max(ratios_eager) if ratios_eager else 0.0
        avg_g = sum(ratios_graph) / max(1, len(ratios_graph)) if ratios_graph else 0.0
        gmin = min(ratios_graph) if ratios_graph else 0.0
        gmax = max(ratios_graph) if ratios_graph else 0.0
        avg_tcg = sum(ratios_trt_cg_over_graph) / max(1, len(ratios_trt_cg_over_graph)) if ratios_trt_cg_over_graph else 0.0
        tcg_min = min(ratios_trt_cg_over_graph) if ratios_trt_cg_over_graph else 0.0
        tcg_max = max(ratios_trt_cg_over_graph) if ratios_trt_cg_over_graph else 0.0
        t_speed = sum(trt_cg_over_trt) / max(1, len(trt_cg_over_trt)) if trt_cg_over_trt else 0.0
        ge = sum(vllm_graph_over_eager) / max(1, len(vllm_graph_over_eager)) if vllm_graph_over_eager else 0.0
        lines = [
            "Key takeaways:",
            f"- more preset avg TRT/vLLM eager (decode tok/s): {avg_e:.4f}  (min={emin:.4f}, max={emax:.4f})",
            f"- more preset avg TRT/vLLM graph (decode tok/s): {avg_g:.4f}  (min={gmin:.4f}, max={gmax:.4f})",
            f"- more preset avg TRT_cg/vLLM graph (decode tok/s): {avg_tcg:.4f}  (min={tcg_min:.4f}, max={tcg_max:.4f})",
            f"- more preset avg TRT_cg/TRT speedup: {t_speed:.4f}",
            f"- more preset avg vLLM graph/eager speedup: {ge:.4f}",
            "- Free-run greedy token_match can be low due to divergence amplification.",
            "- Teacher-forced next-token match better reflects agreement under the same history.",
            "",
            "Added 'simple-level' evidence:",
            "- bs4_p64_s128 seed sweep: token_match / teacher-forced match / min prefix",
            "- Prefill-only next-token sweep: prefill_len=16..512",
            "- Perf variance (mean±std) across TRT/TRT-cg/vLLM-eager/vLLM-graph",
            "",
            f"Artifacts directory: {os.path.relpath(a, os.getcwd())}",
        ]
        _page_text(pdf, "orchid Benchmarks Final Report", lines)

        _page_image(pdf, "Suite Overview (more preset)", os.path.join(a, "simple_suite_more.png"))
        _page_image(pdf, "bs4_p64_s128 Seed Sweep", os.path.join(a, "bs4_p64_s128_seed_sweep.png"))
        _page_image(pdf, "Prefill-only Next-token Sweep", os.path.join(a, "prefill_only_sweep.png"))
        _page_image(pdf, "Perf Variance (mean ± std)", os.path.join(a, "perf_variance.png"))

        if seed:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.02, 0.97, "bs4_p64_s128 Quick Readout", fontsize=16, fontweight="bold", va="top")
            y = 0.92
            ax.text(0.03, y, f"seeds: {len(seed)}", fontsize=12, va="top")
            y -= 0.03
            ax.text(0.03, y, f"token_match_rate mean={sum(seed_tm)/max(1,len(seed_tm)):.4f} min={min(seed_tm):.4f}", fontsize=12, va="top")
            y -= 0.03
            if seed_tf and any(x > 0 for x in seed_tf):
                ax.text(0.03, y, f"tf_next_token_match_rate mean={sum(seed_tf)/max(1,len(seed_tf)):.4f} min={min(seed_tf):.4f}", fontsize=12, va="top")
                y -= 0.03
            if seed_pmin:
                ax.text(0.03, y, f"min prefix length: min={min(seed_pmin)} p50={sorted(seed_pmin)[len(seed_pmin)//2]} max={max(seed_pmin)}", fontsize=12, va="top")
            pdf.savefig(fig)
            plt.close(fig)

    print(out_pdf, flush=True)


if __name__ == "__main__":
    main()
