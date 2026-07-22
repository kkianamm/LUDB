# Composing SCA · MERGETUNE · DGA on top of IPL (ECG-BiomedCLIP)

This adds three optional adaptation modules **on top of your existing IPL base**,
without replacing it. IPL stays the classifier (learned text prompt on the
frozen PubMedBERT tower); the new methods hook the shared BiomedCLIP pipeline at
three *different* stages, which is exactly what makes them stack.

## Why these three compose (the hook taxonomy)

```
 pixels ──(DGA: learnable VR program)──► [FROZEN vision] ──► image features
                                                                   │
                          text features ◄── [FROZEN text + IPL prompt]      (IPL, MERGETUNE)
                                                                   │
                              logits = scale · cos(img, text)
                                                                   │
              loss = BCE + IPL.anchor + DGA.align + MERGETUNE.lmc  (train stages)
                                                                   │
                        test-time logit refinement (SCA)           (inference)
```

| Method | Stage it touches | Modality | Trains? | Breaks feature cache? |
|---|---|---|---|---|
| **IPL** (base) | text prompt + loss | text | yes | no |
| **SCA** | test-time logits | — | no (training-free) | no |
| **MERGETUNE** | text-feature space, post-hoc | text | yes (stage 2) | no |
| **DGA** | input pixels + vision features | vision | yes (stage 1) | **yes** |

Because SCA is inference-only, MERGETUNE is a text-space continued-tuning stage,
and DGA is a vision-side reprogramming, none of them overwrites IPL's prompt.
DGA in its home repo is a *full* pipeline with its own text side — here we take
only its **vision-side** parts (VR + multi-granularity) and align them against
**IPL's** learned prompt, so DGA augments IPL rather than replacing it.

## Files

```
modules/base.py             AdaptationModule interface + ModuleStack orchestrator
modules/registry.py         name -> module, build_stack(exp_cfg, ...)
modules/sca_module.py       SCA: ridge statistics cache, test-time (multi-label)
modules/mergetune_module.py MERGETUNE: LMC line-loss continued fine-tuning
modules/dga_module.py       DGA: PaddingVR + multi-granularity fusion (vision-side)
config_experiments.py       all 8 IPL combos + baselines as flag sets
train_compose.py            one trainer: caching gate + stage-1/stage-2 + eval
```

Drop next to your existing `config_ipl.py`, `train_ipl.py`, and the `ipl/`
package. Nothing in the base repo is modified (same philosophy as `README_IPL.md`).

## Running every required experiment

```bash
# ---- the 8 IPL-based experiments ----
python train_compose.py --exp ipl                    --shots 0   # 1. IPL only
python train_compose.py --exp ipl_sca                --shots 0   # 2. IPL + SCA
python train_compose.py --exp ipl_mergetune          --shots 0   # 3. IPL + MERGETUNE
python train_compose.py --exp ipl_dga                --shots 0   # 4. IPL + DGA
python train_compose.py --exp ipl_sca_mergetune      --shots 0   # 5.
python train_compose.py --exp ipl_sca_dga            --shots 0   # 6.
python train_compose.py --exp ipl_mergetune_dga      --shots 0   # 7.
python train_compose.py --exp ipl_sca_mergetune_dga  --shots 0   # 8. everything

# ---- independent baselines ----
python train_compose.py --exp zeroshot               --shots 0   # zero-shot BiomedCLIP
python train_compose.py --exp sca_only               --shots 0   # SCA without IPL
python train_compose.py --exp dga_only               --shots 0   # DGA without IPL
python train_compose.py --exp mergetune_coop         --shots 0   # MERGETUNE on Phase-A CoOp
```

Few-shot is `--shots 1/2/4/8/16`. Everything routes through the same eval
harness (macro AUROC / per-class AUROC / F1) so results are directly comparable.

## Orchestration details

* **Caching gate.** `train_compose.py` computes `raw = stack.wants_raw_pixels`.
  DGA sets that True, so DGA runs train/eval on raw images (recomputing features
  through the frozen tower each step). Every DGA-free experiment keeps IPL's fast
  cached-feature path untouched. This is the single most important interaction:
  you cannot pre-cache image features when a learnable pixel program is in front
  of the encoder.
* **Two training stages.** Stage 1 trains IPL and DGA jointly (their losses are
  summed on the fused logits). Stage 2 is MERGETUNE's post-hoc continued
  fine-tuning of the *same* IPL params — it runs only after stage 1, matching the
  paper's "continue fine-tuning an already-adapted model" design.
* **Inference composition.** DGA's multi-granularity fusion produces the base
  logits; SCA then wraps the whole eval loop as the outermost refiner. Order is
  fixed in `ModuleStack.refine_test_logits`.

## Honest methodological caveats (read before trusting numbers)

These are the places where the source methods assume single-label / natural
images and had to be adapted for multi-label ECG-on-plots. Each is called out in
the module docstring too.

1. **SCA is single-label in the original.** It uses softmax entropy, argmax
   top-1, and a per-class memory keyed by pseudo-label. The port replaces softmax
   with per-class **sigmoid** and class-entropy with **mean binary entropy**. The
   ridge cache `W = (G+ridge·I)⁻¹Pᵀ` is a linear feats→score map and is
   label-agnostic, so it transfers cleanly. **Tune `ridge`** (start 1e4) and
   note this is *transductive* TTA over the test fold.
2. **SCA × IPL Phase C.** SCA needs a fixed `(C,D)` classifier. When IPL's
   meta-net (Phase C) is on, the classifier is per-image. The eval path feeds SCA
   the **shared** prompt (`prompt_learner(image_features=None)`), i.e. the
   instance-conditioning is used for the base logit but SCA's cache is built on
   the shared classifier. If you want SCA to see instance-conditioned text,
   that's a research change, not a drop-in.
3. **MERGETUNE needs a fine-tuned endpoint.** Endpoints are zero-shot BiomedCLIP
   text (ŵ₁, already stored as `IPLModel.text_anchors`) and the IPL solution
   (ŵ₂). "MERGETUNE without IPL" is therefore only meaningful on *some* fine-tuned
   base — provided as `mergetune_coop` (Phase-A CoOp). A "MERGETUNE on pure
   zero-shot" run has nothing to merge and is intentionally omitted. The released
   code interpolates in text-feature space with a task-loss line integral; the
   paper's zero-shot **second-order surrogate** (to avoid data replay) is *not*
   reproduced here — the zero-shot endpoint is available directly, so the plain
   line-loss is the faithful multi-label analogue.
4. **DGA hierarchy / HKP is single-label.** The Semantic-Granularity branch and
   `HKP` consistency assume a softmax class tree. For 5 PTB-XL superclasses this
   is shallow and multi-label, so `use_semantic_hierarchy` defaults **False** and
   raises if switched on before you re-derive HKP for multi-label. The
   **vision-granularity** part (VR + multi-scale + entropy fusion + BCE against
   IPL text) is the part that ports cleanly and is what runs by default.
5. **DGA normalisation.** The original `PaddingVR` hard-codes OpenAI-CLIP
   mean/std. The port swaps in BiomedCLIP stats and expects **un-normalised
   [0,1]** pixels (see `raw_preprocess`) so the program is added in true pixel
   space. Don't double-normalise.
6. **`dga_only` baseline** here aligns VR against IPL's *untrained* prompt (≈ your
   `CTX_INIT` hand prompt) rather than DGA's native attribute text. For the exact
   published-style DGA number, run the original `iLearn-Lab/CVPR26-DGA` repo; this
   baseline is the apples-to-apples "DGA vision-side, our text" control.
7. **Ceiling.** As your base README notes, rendering a 1-D ECG to a 224px plot
   caps achievable AUROC below purpose-built 1-D models. These modules adapt
   BiomedCLIP better; they don't lift that ceiling.

## Extending

* Add a module by subclassing `AdaptationModule`, setting its capability flags
  (`wants_raw_pixels` / `train_stage` / `is_test_time`), implementing the hooks it
  needs, and registering it in `modules/registry.REGISTRY`. The trainer and eval
  pick it up automatically; new combinations are just new entries in
  `config_experiments.EXPERIMENTS`.
* The exact seams to the source repos are cited at the top of each module file
  (function/class names in SCA, MERGETUNE, DGA) so you can pull in more of each
  method (e.g. DGA's HKP, MERGETUNE's second-order surrogate) at those points.
```
