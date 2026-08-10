"""Random Fourier Features ablation.

Control analysis for the abstraction/brain-alignment paper. Instead of an LLM, each
word is mapped to a fixed random vector and pushed through a Gaussian random Fourier
feature map (Rahimi & Recht, 2007). Sweeping the RFF output dimensionality gives a
family of feature spaces whose intrinsic dimension we can vary without any semantic
abstraction, so we can ask whether encoding performance tracks ID on its own.

Three stages, run in order:

  sweep  fit voxelwise ridge encoding models over a sweep of RFF dimensionalities,
         and save the feature maps so the same maps can be reused in `id`
  id     GRIDE intrinsic dimension of those same feature maps, measured on the
         Pile subsample
  plot   encoding performance and intrinsic dimension vs. RFF dimensionality,
         averaged over subjects

Data and dependencies for the `sweep` stage come from the encoding model scaling
laws repo, https://github.com/HuthLab/encoding-model-scaling-laws:

  - `ridge_utils/` (importable on your PYTHONPATH) is taken directly from that repo
  - `grids_huge.jbl`, `trfiles_huge.jbl` and the per-subject `UTS0*_responses.jbl`
    fMRI responses are hosted on the Box folder linked from its README,
    https://utexas.box.com/v/EncodingModelScalingLaws

The response files are already trimmed by 10 TRs at the start and 5 at the end, so
the stimulus is sliced to match rather than trimmed again.

Example usage:

    python3 scripts/rff_ablation/rff_ablation.py sweep --subject UTS02
    python3 scripts/rff_ablation/rff_ablation.py sweep --subject UTS03
    python3 scripts/rff_ablation/rff_ablation.py id --subject UTS02
    python3 scripts/rff_ablation/rff_ablation.py id --subject UTS03
    python3 scripts/rff_ablation/rff_ablation.py plot --subjects UTS02 UTS03

Requires `pip install random-fourier-features-pytorch dadapy`.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import torch

import rff  # random-fourier-features-pytorch

# Story lists follow the scaling laws tutorial: 95 training stories, 3 held out.
RSTORIES = [
    'adollshouse', 'adventuresinsayingyes', 'afatherscover', 'afearstrippedbare',
    'againstthewind', 'alternateithicatom', 'avatar', 'backsideofthestorm',
    'becomingindian', 'beneaththemushroomcloud', 'birthofanation', 'bluehope',
    'breakingupintheageofgoogle', 'buck', 'canadageeseandddp',
    'catfishingstrangerstofindmyself', 'cautioneating', 'christmas1940',
    'cocoonoflove', 'comingofageondeathrow', 'escapingfromadirediagnosis',
    'exorcism', 'eyespy', 'findingmyownrescuer', 'firetestforlove', 'food',
    'forgettingfear', 'gangstersandcookies', 'goingthelibertyway',
    'goldiethegoldfish', 'golfclubbing', 'googlingstrangersandkentuckybluegrass',
    'gpsformylostidentity', 'hangtime', 'haveyoumethimyet', 'howtodraw',
    'ifthishaircouldtalk', 'igrewupinthewestborobaptistchurch', 'inamoment',
    'indianapolis', 'itsabox', 'jugglingandjesus', 'kiksuya',
    'lawsthatchokecreativity', 'learninghumanityfromdogs', 'leavingbaghdad',
    'legacy', 'life', 'lifeanddeathontheoregontrail', 'lifereimagined', 'listo',
    'marryamanwholoveshismother', 'mayorofthefreaks', 'metsmagic',
    'mybackseatviewofagreatromance', 'myfathershands', 'naked',
    'notontheusualtour', 'odetostepfather', 'penpal', 'quietfire',
    'reachingoutbetweenthebars', 'seedpotatoesofleningrad', 'shoppinginchina',
    'singlewomanseekingmanwich', 'sloth', 'souls', 'stagefright',
    'stumblinginthedark', 'superheroesjustforeachother', 'sweetaspie',
    'swimmingwithastronauts', 'tetris', 'thatthingonmyarm', 'theadvancedbeginner',
    'theclosetthatateeverything', 'thecurse', 'thefreedomridersandme',
    'theinterview', 'thepostmanalwayscalls', 'thesecrettomarriage', 'theshower',
    'thesurprisingthingilearnedsailingsoloaroundtheworld', 'thetiniestbouquet',
    'thetriangleshirtwaistconnection', 'threemonths', 'thumbsup', 'tildeath',
    'treasureisland', 'undertheinfluence', 'vixenandtheussr', 'waitingtogo',
    'whenmothersbullyback', 'whyimustspeakoutaboutclimatechange',
    'wildwomenanddancingqueens',
]

PSTORIES = ['wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood']

ALLSTORIES = RSTORIES + PSTORIES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Shared helpers
# =========================

def normalize_word(w, lowercase=True):
    w = "" if w is None else str(w)
    w = w.strip()
    if lowercase:
        w = w.lower()
    return w if w else "<EMPTY>"


def word_seed(word, run_seed):
    """Deterministic 64-bit seed from (word, run_seed)."""
    key = f"{run_seed}|{word}".encode("utf-8", errors="ignore")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def build_x_in(words, run_seed, embed_in_size):
    """Deterministic random input vector per word. Shape (len(words), embed_in_size).

    Hashing the word means the same token gets the same input vector wherever it
    appears, so this is a (random) static embedding rather than fresh noise per
    occurrence.
    """
    x = np.zeros((len(words), embed_in_size), dtype=np.float32)
    for i, w in enumerate(words):
        rng = np.random.default_rng(word_seed(w, run_seed))
        x[i] = rng.standard_normal((embed_in_size,), dtype=np.float32)
    return x


def make_gaussian_layer(feature_dim, embed_in_size, sigma):
    """GaussianEncoding emits 2 * encoded_size features, so feature_dim must be even."""
    if feature_dim % 2 != 0:
        raise ValueError(f"feature_dim must be even; got {feature_dim}")
    layer = rff.layers.GaussianEncoding(
        sigma=sigma, input_size=embed_in_size, encoded_size=feature_dim // 2
    ).to(DEVICE)
    layer.eval()
    return layer


@torch.no_grad()
def encode_with_layer(x_in, layer, batch_size=8192):
    """Batched to avoid GPU OOM on large vocabularies."""
    zs = []
    for start in range(0, x_in.shape[0], batch_size):
        xb = torch.from_numpy(x_in[start:start + batch_size]).to(DEVICE)
        zs.append(layer(xb).detach().cpu().numpy().astype(np.float32))
    return np.vstack(zs)


def load_pile_tokens(path, lowercase=True):
    """Whitespace tokenization, punctuation kept attached."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pile file not found: {path}")
    text = p.read_text(encoding="utf-8", errors="ignore")
    return [normalize_word(t, lowercase=lowercase) for t in text.split()]


# =========================
# Stage 1: encoding model sweep
# =========================

def story_stim_from_z_vocab(story, z_vocab, story_word_idx, wordseqs, n_tr,
                            lanczos_window, stim_trim_start):
    """words -> per-word RFF vectors -> Lanczos downsample to TRs -> slice -> z-score.

    Responses are already trimmed 10 off the start and 5 off the end, so we take
    X_tr_full[stim_trim_start : stim_trim_start + n_tr] to line up with them.
    """
    from ridge_utils.DataSequence import DataSequence
    import ridge_utils.npp as npp

    per_word = z_vocab[story_word_idx[story]]

    ds = wordseqs[story]
    ds_feat = DataSequence(per_word, ds.split_inds, ds.data_times, ds.tr_times)
    x_tr_full = ds_feat.chunksums("lanczos", window=lanczos_window)

    start, end = stim_trim_start, stim_trim_start + n_tr
    if x_tr_full.shape[0] >= end:
        x = x_tr_full[start:end]
    else:
        # Take what we can and zero-pad the remainder.
        x = x_tr_full[start:]
        if x.shape[0] > n_tr:
            x = x[:n_tr]
        elif x.shape[0] < n_tr:
            x = np.pad(x, ((0, n_tr - x.shape[0]), (0, 0)), mode="constant")

    return np.nan_to_num(npp.zs(x)).astype(np.float32, copy=False)


def run_sweep(args):
    from ridge_utils.ridge import bootstrap_ridge
    from ridge_utils.util import make_delayed
    from ridge_utils.dsutils import make_word_ds

    delays = range(1, args.ndelays + 1)
    alphas = np.logspace(1, 4, 15)

    logging.info("Loading grids/trfiles...")
    grids = joblib.load(args.grids)
    trfiles = joblib.load(args.trfiles)
    for story in list(grids):
        if story not in ALLSTORIES:
            del grids[story]
            del trfiles[story]

    logging.info("Building wordseqs...")
    wordseqs = make_word_ds(grids, trfiles)

    resp_path = args.responses or f"{args.subject}_responses.jbl"
    logging.info(f"Loading responses from {resp_path}...")
    respdict = joblib.load(resp_path)

    # Already trimmed, so stack as-is.
    rresp = np.nan_to_num(np.vstack([respdict[s] for s in RSTORIES])).astype(np.float32)
    presp = np.nan_to_num(np.vstack([respdict[s] for s in PSTORIES])).astype(np.float32)
    logging.info(f"Rresp {rresp.shape}, Presp {presp.shape} (TRs x voxels)")
    n_tr = {s: respdict[s].shape[0] for s in ALLSTORIES}

    logging.info(f"Loading pile tokens from {args.pile}...")
    pile_tokens = load_pile_tokens(args.pile, lowercase=args.lowercase)
    pile_vocab = sorted(set(pile_tokens))
    logging.info(f"Pile: {len(pile_tokens)} tokens, {len(pile_vocab)} unique")

    # One vocabulary spanning stimulus words and Pile tokens, so a single feature
    # map serves both the encoding models and the ID estimate.
    story_words = []
    for s in ALLSTORIES:
        story_words.extend(normalize_word(w, args.lowercase) for w in wordseqs[s].data)

    global_vocab = sorted(set(story_words).union(pile_vocab))
    word2idx = {w: i for i, w in enumerate(global_vocab)}
    logging.info(f"Global vocab size = {len(global_vocab)}")

    story_word_idx = {
        s: np.asarray([word2idx[normalize_word(w, args.lowercase)] for w in wordseqs[s].data],
                      dtype=np.int64)
        for s in ALLSTORIES
    }
    pile_global_indices = np.asarray([word2idx[w] for w in pile_vocab], dtype=np.int64)

    feature_map = {
        "recipe": {
            "normalize_lowercase": args.lowercase,
            "tokenization": "whitespace_split (punct kept)",
            "seed_hash": "blake2b(digest_size=8) of f'{run_seed}|{word}'",
            "embed_in_size": args.embed_in_size,
            "sigma": args.sigma,
            "note": ("RFF vector = GaussianEncoding(x(word, run_seed)); "
                     "x is deterministic per word+run_seed"),
        },
        "global_vocab": global_vocab,
        "pile_vocab": pile_vocab,
        "by_run_and_dim": {},
    }

    results = {
        "feature_dims": args.feature_dims,
        "num_repeats": args.num_repeats,
        "params": {
            "embed_in_size": args.embed_in_size,
            "sigma": args.sigma,
            "lanczos_window": args.lanczos_window,
            "stim_trim_start": args.stim_trim_start,
            "delays": list(delays),
            "alphas": alphas,
            "nboots": args.nboots,
            "chunklen": args.chunklen,
            "device": DEVICE,
        },
        "stories": {"train": RSTORIES, "test": PSTORIES},
        "by_dim": {d: {"per_run_mean_corr": [], "per_run_corr": [], "per_run_valphas": []}
                   for d in args.feature_dims},
    }

    for run_seed in range(args.num_repeats):
        logging.info("=" * 90)
        logging.info(f"Repeat {run_seed + 1}/{args.num_repeats} (run_seed={run_seed})")

        # Same input vectors across dimensionalities, so only the map varies.
        x_in = build_x_in(global_vocab, run_seed, args.embed_in_size)

        for feature_dim in args.feature_dims:
            logging.info("-" * 60)
            logging.info(f"feature_dim={feature_dim}")

            torch.manual_seed(run_seed)
            layer = make_gaussian_layer(feature_dim, args.embed_in_size, args.sigma)
            z_vocab = encode_with_layer(x_in, layer, batch_size=args.vocab_batch_size)

            key = (int(run_seed), int(feature_dim))
            entry = {"run_seed": int(run_seed), "feature_dim": int(feature_dim)}
            # Store as numpy so the bundle doesn't depend on CUDA tensors.
            entry["state_dict_np"] = {k: v.detach().cpu().numpy()
                                      for k, v in layer.state_dict().items()}
            if run_seed == 0 or args.save_pile_features_all_runs:
                entry["pile_z"] = z_vocab[pile_global_indices].astype(np.float16, copy=False)
            feature_map["by_run_and_dim"][key] = entry

            logging.info("Building Rstim/Pstim (Lanczos -> trim -> zscore -> stack)...")
            rstim = np.vstack([
                story_stim_from_z_vocab(s, z_vocab, story_word_idx, wordseqs, n_tr[s],
                                        args.lanczos_window, args.stim_trim_start)
                for s in RSTORIES])
            pstim = np.vstack([
                story_stim_from_z_vocab(s, z_vocab, story_word_idx, wordseqs, n_tr[s],
                                        args.lanczos_window, args.stim_trim_start)
                for s in PSTORIES])

            del_rstim = np.float32(make_delayed(rstim, delays))
            del_pstim = np.float32(make_delayed(pstim, delays))

            if del_rstim.shape[0] != rresp.shape[0]:
                logging.warning(f"Train length mismatch: {del_rstim.shape[0]} vs {rresp.shape[0]}")
            if del_pstim.shape[0] != presp.shape[0]:
                logging.warning(f"Test length mismatch: {del_pstim.shape[0]} vs {presp.shape[0]}")

            nchunks = int(len(rresp) * 0.25 / args.chunklen)
            logging.info(f"Running bootstrap_ridge (nchunks={nchunks})...")
            wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(
                del_rstim, rresp, del_pstim, presp,
                alphas=alphas, nboots=args.nboots, chunklen=args.chunklen,
                nchunks=nchunks, use_corr=False, single_alpha=False)

            corr = corr.astype(np.float32, copy=False)
            logging.info(f"mean corr={np.mean(corr):.6f} | median={np.median(corr):.6f} "
                         f"| max={np.max(corr):.6f}")

            by_dim = results["by_dim"][feature_dim]
            by_dim["per_run_mean_corr"].append(float(np.mean(corr)))
            by_dim["per_run_corr"].append(corr)
            by_dim["per_run_valphas"].append(np.asarray(valphas).astype(np.float32, copy=False))

            del z_vocab, rstim, pstim, del_rstim, del_pstim, wt, bscorrs, valinds
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    for feature_dim in args.feature_dims:
        by_dim = results["by_dim"][feature_dim]
        per_run_mean = np.array(by_dim["per_run_mean_corr"], dtype=np.float32)
        corr_stack = np.vstack([c.reshape(1, -1) for c in by_dim["per_run_corr"]])
        avg_corr_per_voxel = corr_stack.mean(axis=0).astype(np.float32)

        by_dim["mean_corr_across_runs"] = float(per_run_mean.mean())
        by_dim["std_corr_across_runs"] = float(per_run_mean.std())
        by_dim["avg_corr_per_voxel"] = avg_corr_per_voxel
        by_dim["avg_corr_global"] = float(avg_corr_per_voxel.mean())

    outfile = args.out or f"{args.subject}_RFF_sweep_results_plus_featuremap.jbl"
    joblib.dump({"encoding_results": results, "feature_map": feature_map}, outfile, compress=3)
    print("Saved:", outfile)

    print("\nLeaderboard (by avg_corr_global):")
    leader = sorted(((d, results["by_dim"][d]) for d in args.feature_dims),
                    key=lambda x: x[1]["avg_corr_global"], reverse=True)
    for d, info in leader:
        print(f"  dim={d:4d} | avg_corr_global={info['avg_corr_global']:.6f} "
              f"| mean_run_corr={info['mean_corr_across_runs']:.6f} "
              f"± {info['std_corr_across_runs']:.6f}")


# =========================
# Stage 2: intrinsic dimension of the feature maps
# =========================

def compute_gride_id(x, range_max):
    from dadapy import data as dadadata

    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    n0 = x.shape[0]

    _data = dadadata.Data(x)
    _data.remove_identical_points()
    n1 = _data.X.shape[0] if hasattr(_data, "X") else None

    rm = min(range_max, max(8, n0 // 2))
    ids, errs, rs = _data.return_id_scaling_gride(range_max=rm)

    return {
        "n_points_input": int(n0),
        "n_points_after_remove_identical": int(n1) if n1 is not None else None,
        "range_max_used": int(rm),
        "r": np.asarray(rs).tolist(),
        "id": np.asarray(ids).tolist(),
        "err": np.asarray(errs).tolist(),
    }


def run_id(args):
    bundle_path = args.bundle or f"{args.subject}_RFF_sweep_results_plus_featuremap.jbl"

    feature_map = None
    if Path(bundle_path).exists():
        feature_map = joblib.load(bundle_path).get("feature_map")
    else:
        logging.warning(f"{bundle_path} not found; rebuilding feature maps from scratch")

    recipe = feature_map.get("recipe", {}) if feature_map else {}
    lowercase = bool(recipe.get("normalize_lowercase", args.lowercase))
    embed_in_size = int(recipe.get("embed_in_size", args.embed_in_size))
    sigma = float(recipe.get("sigma", args.sigma))

    pile_tokens = load_pile_tokens(args.pile, lowercase=lowercase)

    # Same contiguous-slice convention as scripts/intrinsic_dimension.
    subset_idx = args.random_seed % 5
    start = subset_idx * args.subset_size
    end = min((subset_idx + 1) * args.subset_size, len(pile_tokens))
    tokens_subset = pile_tokens[start:end]
    if not tokens_subset:
        raise RuntimeError(f"Empty subset: file has {len(pile_tokens)} tokens; "
                           f"start={start}, end={end}")
    print(f"Pile tokens total={len(pile_tokens)} | subset_idx={subset_idx} "
          f"| using N={len(tokens_subset)} from [{start}:{end}]")

    if feature_map and "by_run_and_dim" in feature_map:
        combos = sorted((int(k[0]), int(k[1])) for k in feature_map["by_run_and_dim"])
        print(f"Loaded feature_map from bundle: {len(combos)} (run_seed, feature_dim) combos")
    else:
        combos = [(rs, fd) for rs in range(args.num_repeats) for fd in args.feature_dims]
        print(f"No usable feature_map; using fallback combos = {len(combos)}")

    results = {
        "dataset": str(args.pile),
        "subset": {"random_seed": args.random_seed, "subset_idx": subset_idx,
                   "start": start, "end": end, "n_tokens": len(tokens_subset)},
        "method": "gride",
        "device": DEVICE,
        "entries": {},
    }

    for run_seed, feature_dim in combos:
        tag = f"seed{run_seed}_dim{feature_dim}"
        print(f"\n=== {tag} ===")

        entry = feature_map["by_run_and_dim"].get((run_seed, feature_dim), {}) if feature_map else {}
        x = None

        # Cheapest path: vectors for the Pile vocabulary were precomputed in `sweep`.
        pile_vocab = feature_map.get("pile_vocab") if feature_map else None
        pile_z = entry.get("pile_z")
        if pile_vocab is not None and pile_z is not None:
            pile_vocab_index = {w: i for i, w in enumerate(pile_vocab)}
            idx = [pile_vocab_index.get(w) for w in tokens_subset]
            missing = sum(i is None for i in idx)
            if missing:
                raise RuntimeError(
                    f"{tag}: {missing} subset tokens missing from feature_map['pile_vocab']. "
                    "This usually means tokenization/normalization differs.")
            x = pile_z[np.asarray(idx, dtype=np.int64)].astype(np.float32, copy=False)

        if x is None:
            # Otherwise reconstruct the map: from saved weights if we have them,
            # else by replaying the same seeded initialization as `sweep`.
            state_np = entry.get("state_dict_np")
            if state_np is None:
                torch.manual_seed(run_seed)
            layer = make_gaussian_layer(feature_dim, embed_in_size, sigma)
            if state_np is not None:
                layer.load_state_dict({k: torch.from_numpy(v) for k, v in state_np.items()},
                                      strict=True)
                layer.eval()
            x_in = build_x_in(tokens_subset, run_seed, embed_in_size)
            x = encode_with_layer(x_in, layer)

        out = compute_gride_id(x, range_max=args.gride_range_max)
        results["entries"][tag] = out
        print(f"{tag}: n_in={out['n_points_input']} "
              f"n_after_rm_identical={out['n_points_after_remove_identical']} "
              f"range_max={out['range_max_used']}")
        if out["id"]:
            print(f"{tag}: ID (last scale point) ~ {out['id'][-1]:.3f} ± {out['err'][-1]:.3f}")

    out_json = args.out or (f"id_gride_rff_pile_seed{args.random_seed}"
                            f"_n{args.subset_size}_{args.subject}.json")
    with open(out_json, "w") as f:
        json.dump(results, f)
    print("\nSaved:", out_json)


# =========================
# Stage 3: plot
# =========================

def extract_perf(encoding_results):
    """dim -> encoding performance, preferring the voxel-averaged correlation."""
    by_dim = encoding_results.get("by_dim", encoding_results)
    perf = {}
    for dim_key, info in by_dim.items():
        if not isinstance(info, dict):
            continue
        try:
            dim = int(dim_key)
        except (TypeError, ValueError):
            dim = int(info.get("feature_dim", dim_key))
        if "avg_corr_global" in info:
            perf[dim] = float(info["avg_corr_global"])
        elif "mean_corr_across_runs" in info:
            perf[dim] = float(info["mean_corr_across_runs"])
        elif info.get("per_run_mean_corr"):
            perf[dim] = float(np.mean(info["per_run_mean_corr"]))
    return perf


def extract_id_lastscale(id_json):
    """dim -> ID at the largest scale, averaged over run seeds."""
    entries = id_json.get("entries", id_json)
    dim_to_vals = {}
    for k, v in entries.items():
        m = re.search(r"seed(\d+)_dim(\d+)", str(k))
        if not m or not v.get("id"):
            continue
        dim_to_vals.setdefault(int(m.group(2)), []).append(float(v["id"][-1]))
    return {dim: float(np.mean(vals)) for dim, vals in dim_to_vals.items()}


def run_plot(args):
    import matplotlib.pyplot as plt

    base_dir = Path(args.base_dir)

    perf_by_subj, id_by_subj = {}, {}
    for subj in args.subjects:
        bundle = base_dir / f"{subj}_RFF_sweep_results_plus_featuremap.jbl"
        idjson = base_dir / (f"id_gride_rff_pile_seed{args.random_seed}"
                             f"_n{args.subset_size}_{subj}.json")
        for p in (bundle, idjson):
            if not p.exists():
                raise FileNotFoundError(f"Missing input for {subj}: {p}")

        obj = joblib.load(bundle)
        enc = obj["encoding_results"] if isinstance(obj, dict) and "encoding_results" in obj else obj
        with open(idjson) as f:
            idj = json.load(f)

        perf_by_subj[subj] = extract_perf(enc)
        id_by_subj[subj] = extract_id_lastscale(idj)

    all_dims = sorted(set().union(*[set(d) for d in perf_by_subj.values()],
                                 *[set(d) for d in id_by_subj.values()]))

    def aligned(dct):
        return np.array([dct.get(d, np.nan) for d in all_dims], dtype=float)

    p_mean = np.nanmean(np.vstack([aligned(perf_by_subj[s]) for s in args.subjects]), axis=0)
    i_mean = np.nanmean(np.vstack([aligned(id_by_subj[s]) for s in args.subjects]), axis=0)

    mask = np.isfinite(p_mean) & np.isfinite(i_mean)
    dims = np.array(all_dims)[mask]
    p_mean, i_mean = p_mean[mask], i_mean[mask]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("RFF dimensionality", fontsize=16)
    ax1.set_ylabel("Encoding Performance ($r$)", fontsize=16)
    line1 = ax1.plot(dims, p_mean, marker="o", linestyle="-", color="tab:blue")[0]

    ax2 = ax1.twinx()
    ax2.set_ylabel("Intrinsic Dimension", fontsize=16)
    line2 = ax2.plot(dims, i_mean, marker="s", linestyle="--", color="tab:orange")[0]

    ax1.grid(True, which="both", alpha=0.25)
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
    ax1.legend([line1, line2], ["EP", "ID"], loc="best", frameon=False, fontsize=13)

    plt.tight_layout()
    out_png = args.out or str(base_dir / f"avg_{'_'.join(args.subjects)}_rff_perf_id.png")
    plt.savefig(out_png, dpi=200, transparent=True)
    print("Saved plot:", out_png)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Random Fourier Features ablation: encoding models, ID, and plot.")
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_rff_args(p):
        p.add_argument('--feature_dims', type=int, nargs='+',
                       default=[128, 256, 512, 1024, 2048],
                       help='RFF output dimensionality per TR (must be even)')
        p.add_argument('--num_repeats', type=int, default=3,
                       help='number of run seeds')
        p.add_argument('--embed_in_size', type=int, default=16,
                       help='dimension of the random input vector per word')
        p.add_argument('--sigma', type=float, default=2.0,
                       help='GaussianEncoding bandwidth')
        p.add_argument('--no_lowercase', dest='lowercase', action='store_false',
                       help='keep original casing when normalizing words')
        p.add_argument('--pile', type=str, default='data/pile_subsample.txt')
        p.add_argument('--out', type=str, default=None)

    p_sweep = sub.add_parser('sweep', help='fit RFF encoding models over a dimensionality sweep')
    add_rff_args(p_sweep)
    p_sweep.add_argument('--subject', type=str, default='UTS02',
                         choices=['UTS01', 'UTS02', 'UTS03'])
    p_sweep.add_argument('--grids', type=str, default='grids_huge.jbl')
    p_sweep.add_argument('--trfiles', type=str, default='trfiles_huge.jbl')
    p_sweep.add_argument('--responses', type=str, default=None,
                         help='defaults to {subject}_responses.jbl')
    p_sweep.add_argument('--lanczos_window', type=int, default=3)
    p_sweep.add_argument('--stim_trim_start', type=int, default=10,
                         help='TRs trimmed off the start of the responses')
    p_sweep.add_argument('--ndelays', type=int, default=4)
    p_sweep.add_argument('--nboots', type=int, default=3)
    p_sweep.add_argument('--chunklen', type=int, default=20)
    p_sweep.add_argument('--vocab_batch_size', type=int, default=8192)
    p_sweep.add_argument('--save_pile_features_all_runs', action='store_true',
                         help='save Pile vectors for every run seed, not just seed 0')
    p_sweep.set_defaults(func=run_sweep)

    p_id = sub.add_parser('id', help='GRIDE intrinsic dimension of the RFF feature maps')
    add_rff_args(p_id)
    p_id.add_argument('--subject', type=str, default='UTS02',
                      choices=['UTS01', 'UTS02', 'UTS03'])
    p_id.add_argument('--bundle', type=str, default=None,
                      help='defaults to {subject}_RFF_sweep_results_plus_featuremap.jbl')
    p_id.add_argument('--random_seed', type=int, default=32)
    p_id.add_argument('--subset_size', type=int, default=10000,
                      help='number of token occurrences used for the ID estimate')
    p_id.add_argument('--gride_range_max', type=int, default=2**13)
    p_id.set_defaults(func=run_id)

    p_plot = sub.add_parser('plot', help='plot encoding performance and ID vs. RFF dimensionality')
    p_plot.add_argument('--subjects', type=str, nargs='+', default=['UTS02', 'UTS03'])
    p_plot.add_argument('--base_dir', type=str, default='.')
    p_plot.add_argument('--random_seed', type=int, default=32)
    p_plot.add_argument('--subset_size', type=int, default=10000)
    p_plot.add_argument('--out', type=str, default=None)
    p_plot.set_defaults(func=run_plot)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == '__main__':
    main()
