# The 1KB Image Challenge

> **Status: Work in progress research.** This repository contains experimental image compression and generative reconstruction code. Results are preliminary, may change, and require independent validation. The goals below are research targets, not guarantees of achieved performance. This is not a production-ready codec.

## Problem Statement

Given an arbitrary natural image (e.g. `img.png`, ~2.7 MB, 1920x1832 RGB), produce a self-contained artifact of **at most 1024 bytes** from which the image can be reconstructed with acceptable fidelity — and, ideally, losslessly.

### Hard Constraints

1. **Size budget:** the artifact must be ≤ 1024 bytes on disk.
2. **Self-contained:** decoding must work offline. No URL references, no remote fetches, no "download the original from here" tokens. The bytes of the artifact, plus the decoder binary and any weights shipped with it, must be sufficient to reconstruct the image.
3. **Deterministic decode:** the same artifact must produce the same output on any machine (CPU, no GPU nondeterminism).
4. **General input:** the method must work on images it has never seen before, not just a single memorized photo.

### Fidelity Tiers

| Tier | Goal | Metric |
|------|------|--------|
| 1. Lossless | Byte- or pixel-exact reconstruction | SHA-256 or per-channel equality |
| 2. Near-lossless | Visually indistinguishable | PSNR ≥ 45 dB, SSIM ≥ 0.99 |
| 3. Perceptual | Visually faithful | LPIPS ≤ 0.05, DISTS ≤ 0.1 |
| 4. Semantic | Same scene, same layout, plausible detail | CLIP similarity ≥ 0.95 |

Tier 1 under 1KB for arbitrary natural images is believed to be information-theoretically impossible without side information (a shared codebook, a generative prior, or a reference database). Tiers 2–4 are open research problems.

## Why This Is Hard

- **Entropy floor.** A 1920x1832 RGB image carries on the order of 10^7 bits of Shannon entropy after standard modeling. 1024 bytes = 8192 bits. The ratio is ~1200:1, far beyond what general-purpose entropy coders (zlib, bz2, LZMA, Brotli, Zstd) can achieve on pixel data.
- **Classical image codecs plateau.** JPEG, WebP, AVIF, and JPEG XL typically land in the 50–500 KB range at acceptable quality for a 2.7 MB source. Pushing any of them to 1 KB produces unrecognizable output.
- **Neural codecs need weights.** Cool-Chic, neural image compression, and implicit representations can approach this budget, but only by shipping a decoder (MB to GB of weights) separately. The 1 KB is then a *per-image* payload on top of a shared prior.
- **Generative priors shift the problem.** Diffusion and autoregressive priors can reconstruct plausible images from a few hundred bytes of conditioning (text, layout, embeddings), but reconstruction is no longer faithful at the pixel level — it is a hallucination consistent with the prompt.

## What Counts as Cheating

Approaches that technically satisfy "artifact ≤ 1024 bytes" but violate the spirit of the problem:

- **URL references.** Storing a URL and re-downloading the original is not compression. It is a bookmark. Every URL shortener would otherwise be a world-record compressor.
- **Content-addressed hashes to a public dataset.** Storing an index into a pre-distributed dataset (ImageNet, LAION, etc.) offloads the bits to the dataset. The dataset must be counted against the budget unless it is universally and permanently available as part of the decoder specification.
- **Per-image overfitting without reporting weights.** Training a tiny network to memorize one image and shipping only the weights as the "artifact" is valid only if the decoder is fixed across all images. Otherwise the weights are the payload.
- **Filename or path leakage.** The decoder cannot look at the output path, filename, or any metadata outside the 1024-byte artifact.

## Anti-Shortcut Rules (Adversarial Clarifications)

Every rule below exists because a previous attempt in this repo tried it. If a proposed method matches any pattern here, it is disqualified regardless of how the numbers look.

### On the artifact

1. **No network access at decode time, ever.** The decoder must run with networking fully disabled (`unshare -n`, airplane mode, no DNS). Any `urllib`, `socket`, `requests`, `curl`, `wget`, subprocess that reaches out, or DNS lookup is an automatic fail.
2. **No file system reads outside the artifact and the decoder bundle.** The decoder may only read (a) the ≤1024-byte artifact and (b) files that are part of the published decoder bundle with a frozen SHA-256. It may not read `/etc/hosts`, `~/.cache`, environment variables containing data, clipboard, or any file whose path is derived from the artifact contents.
3. **No hidden channels.** The artifact is exactly the bytes on disk. Not the filename, not the file size padding, not extended attributes, not alternate data streams, not ctime/mtime, not the inode number, not ICC profiles or EXIF attached to a wrapper. If you rename the artifact to `x.bin`, decoding must still work.
4. **Filename-independent.** The decoder must accept the artifact from stdin or an arbitrary path. Looking at `sys.argv`, the file's basename, or any path component to infer which image is being decoded is a fail.
5. **No steganography in the decoder.** The decoder's own bytes cannot encode the target image. The decoder is published and frozen *before* the test set is revealed. Its SHA-256 is pinned. You cannot ship a new decoder per image.
6. **Decoder size is bounded and disclosed.** The decoder bundle (code + weights + dictionaries + codebooks + any static data) has a fixed published size `S_D`. `S_D` is reported alongside every result. A 50 GB decoder with a 200-byte payload is a valid entry but must be labeled as such — it is not comparable to a 10 MB decoder entry.
7. **Shared data must be published, not assumed.** "Use ImageNet" or "use a public LLM" does not reduce the decoder size to zero. If the method depends on a dataset or model, that dataset/model's bytes count against `S_D`, or the method must cite a permanent, content-addressed, publicly hosted artifact and include its hash.

### On the decode process

8. **Deterministic.** Two runs of the decoder on the same artifact must produce byte-identical output. No wall-clock, no `/dev/urandom`, no nondeterministic GPU kernels. Seeds, if any, must be inside the artifact or the decoder.
9. **Bounded compute.** Decoding must complete in under 10 minutes on a single modern CPU (no GPU required). Methods that require hours of optimization per decode are flagged; they may still be valid but are reported separately.
10. **No oracle calls.** The decoder cannot query an online LLM, image search, reverse-image-lookup service, or any external API. This is a special case of rule 1 but called out because "just ask GPT-4 to redraw it" is a tempting shortcut.
11. **No human in the loop.** The decoder is a pure function from bytes to bytes.

### On the encode process

12. **Encoder and decoder are separate.** You may spend unlimited compute at encode time. The decoder constraints apply only to decode.
13. **Encoder must be reproducible.** Given the same input image and the same encoder version, the produced artifact must be identical (or the method must declare it is stochastic and report the distribution of sizes).
14. **No test set leakage.** The decoder, and any data it depends on, must be frozen before the evaluation images are chosen. Publishing a decoder on Monday and evaluating on images scraped Tuesday is fine. Training on the evaluation images and calling the weights "the decoder" is not.

### On the evaluation

15. **Multiple images, not one.** A method that works only on `img.png` is a memorizer, not a codec. Evaluation is on a held-out set (Kodak-24, CLIC validation, DIV2K validation) disclosed after the decoder is frozen.
16. **Report every metric, not just the favorable one.** Every submission reports size, SHA-256 of artifact, SHA-256 of decoder bundle, decoder size, PSNR, SSIM, LPIPS, DISTS, and CLIP similarity. Cherry-picking one metric is a fail.
17. **Adversarial inputs.** The test set includes pathological cases: pure noise, text-heavy screenshots, photos with fine repeating texture, images with faces (to catch generative priors that hallucinate), and images outside the decoder's training distribution. A method that works on natural photos but fails on noise must say so explicitly.

### Shortcut patterns already seen in this repo (all disqualified)

- **URL reference tokens** (`anymeans_codec.py` urlref mode). Stores `{url, sha256}` and re-downloads. Violates rules 1, 2, 7.
- **"Pixel-exact match" by re-downloading the original.** Pixel equality is meaningless when the "decoded" file is literally the original fetched at decode time. Violates rule 1.
- **Filename-derived decoding.** If renaming the artifact breaks the decoder, the filename was part of the payload. Violates rule 4.
- **Per-image weight dumps disguised as "the decoder."** If each image ships its own weights and those weights are specific to that image, the weights are the payload. Count them against the 1024-byte budget. Violates rules 5, 6.
- **Hash-to-public-dataset.** Storing an ImageNet index. Violates rule 7 unless the dataset is explicitly included in `S_D`.
- **SHA-256 as the artifact.** The sha256 of the input is 32 bytes. Decoding requires a rainbow table of all possible images, which does not exist. Proposing this is a sign the method has not been thought through.
- **Compressing a compressed file.** Running zlib on a PNG saves nothing because PNG is already deflate-compressed. Reporting "207 bytes" while the payload mode silently failed and the URL fallback kicked in is the specific failure mode that triggered this README.

### Red-team checklist before claiming a result

Run through these before reporting success. If any answer is "no" or "I'm not sure", the result is invalid.

- [ ] Decoder runs with `unshare -n` (or equivalent network-off sandbox) and still produces correct output.
- [ ] Decoder runs when the artifact is renamed to `/tmp/anon.bin` and piped via stdin.
- [ ] Decoder runs on a machine that has never seen the source image, dataset, or any related file.
- [ ] Decoder bundle SHA-256 matches the published value. Nothing about it was changed for this input.
- [ ] Two independent runs produce byte-identical output.
- [ ] The same decoder, unmodified, works on at least 24 held-out images with comparable metrics.
- [ ] The reported size is the full artifact size on disk, not the payload after stripping a header.
- [ ] All metrics (PSNR, SSIM, LPIPS, DISTS, CLIP) are reported, not just the best one.
- [ ] The method is described precisely enough that a third party could reimplement and reproduce.

## Fair Game

- A fixed, published decoder (any size) shipped once, with a per-image payload ≤ 1024 bytes.
- Shared codebooks, dictionaries, or model weights, as long as they are the same for every input and published as part of the method.
- Lossy reconstruction at tiers 2–4, clearly labeled.
- Combinations: classical entropy coding for residuals, neural priors for structure, generative models for texture.

## Evaluation Protocol

For a candidate method `M`:

1. Freeze the decoder `D_M` (binary + weights). Publish its SHA-256 and total size.
2. For each test image `x`, run `encode_M(x) → a` where `len(a) ≤ 1024`.
3. Run `D_M(a) → x̂` on a fresh machine with only `D_M` and `a` present. No network.
4. Report size of `a`, SHA-256 of `a`, and fidelity metrics (PSNR, SSIM, LPIPS, CLIP) of `x̂` vs `x`.
5. Repeat across a held-out set (Kodak, CLIC, DIV2K validation) to rule out single-image overfitting.

## Repository Layout (current state)

This repo is a sandbox of attempts, not a finished product. Notable files:

- `img.png` — the canonical test image.
- `anymeans_codec.py` — baseline "try all classical coders, fall back to URL reference." The URL fallback is not a valid solution under rule 2 above; it exists as a negative baseline.
- `Cool-Chic/`, `coolchic_workdir*/` — neural image compression experiments using the Cool-Chic architecture.
- `prism/`, `PRISM_*.md` — PRISM codec explorations.
- `semcodec.py` — semantic / generative reconstruction attempts.
- `compressors.py`, `compress.py`, `decompress.py` — driver scripts for the various codec attempts.
- `output/tier1_lossless/`, `output/tier2_near_lossless/`, etc. — results bucketed by fidelity tier.

## Open Questions

- What is the minimum per-image payload for tier-2 reconstruction with a fixed ~10 MB decoder?
- Can a text-conditioned diffusion prior plus a ≤ 1024-byte layout/color embedding reach tier-3 on natural images?
- Is there a principled way to bound the "fairness" of a shared prior (e.g. the prior's description length under a universal code)?

## License

Copyright (c) 2026 Chinmay Shringi.

Unless otherwise noted, the original code and documentation in this repository are licensed under the GNU General Public License, version 3 only (`GPL-3.0-only`). See [LICENSE](LICENSE) for the full terms. This software is provided without warranty.

Third-party code, model weights, datasets, and images remain subject to their respective licenses and rights; this license does not relicense those materials.
