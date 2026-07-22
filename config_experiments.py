"""
Every experiment is just a set of flags over the SAME pipeline. IPL is always
the base; SCA/MERGETUNE/DGA are toggled under `modules`. Nothing here is
mutually exclusive -- adding DGA does not remove IPL, it layers on top.

An experiment dict:
  {
    "ipl_phase": "A"|"B"|"C",          # IPL staging (C = full IPL)
    "modules": {                        # only listed+enabled modules are built
        "sca":       {...} | absent,
        "mergetune": {...} | absent,
        "dga":       {...} | absent,
    },
    "base_only": bool,                  # baselines that skip the IPL prompt (zero-shot / native)
  }
"""

# ---- default per-module hyper-params (override per experiment as needed) ----
SCA_DEFAULTS = dict(enabled=True, ridge=1e4, beta=5.0, tau=0.5)
MERGETUNE_DEFAULTS = dict(enabled=True, w_lmc=1.0, num_samples=10, epochs=10, lr=2e-3)
DGA_DEFAULTS = dict(enabled=True, input_size=192, num_vg_scales=2,
                    lambda_align=1.0, use_semantic_hierarchy=False, lr=40.0)

# =====================================================================
# The 8 required IPL-based experiments
# =====================================================================
EXPERIMENTS = {
    # 1. IPL only
    "ipl": dict(ipl_phase="C", modules={}),

    # 2. IPL + SCA
    "ipl_sca": dict(ipl_phase="C", modules={"sca": dict(SCA_DEFAULTS)}),

    # 3. IPL + MERGETUNE
    "ipl_mergetune": dict(ipl_phase="C", modules={"mergetune": dict(MERGETUNE_DEFAULTS)}),

    # 4. IPL + DGA
    "ipl_dga": dict(ipl_phase="C", modules={"dga": dict(DGA_DEFAULTS)}),

    # 5. IPL + SCA + MERGETUNE
    "ipl_sca_mergetune": dict(ipl_phase="C", modules={
        "sca": dict(SCA_DEFAULTS), "mergetune": dict(MERGETUNE_DEFAULTS)}),

    # 6. IPL + SCA + DGA
    "ipl_sca_dga": dict(ipl_phase="C", modules={
        "sca": dict(SCA_DEFAULTS), "dga": dict(DGA_DEFAULTS)}),

    # 7. IPL + MERGETUNE + DGA
    "ipl_mergetune_dga": dict(ipl_phase="C", modules={
        "mergetune": dict(MERGETUNE_DEFAULTS), "dga": dict(DGA_DEFAULTS)}),

    # 8. IPL + SCA + MERGETUNE + DGA
    "ipl_sca_mergetune_dga": dict(ipl_phase="C", modules={
        "sca": dict(SCA_DEFAULTS),
        "mergetune": dict(MERGETUNE_DEFAULTS),
        "dga": dict(DGA_DEFAULTS)}),

    # =================================================================
    # Independent baselines
    # =================================================================
    # Zero-shot BiomedCLIP (no prompt learning, no modules): use zero_shot_eval.py,
    # or run this to route through the same eval harness.
    "zeroshot": dict(ipl_phase=None, base_only=True, modules={}),

    # SCA without IPL: SCA on top of the zero-shot text classifier (its native setting).
    "sca_only": dict(ipl_phase=None, base_only=True, modules={"sca": dict(SCA_DEFAULTS)}),

    # DGA without IPL: DGA's vision-side pipeline against zero-shot text (native-ish).
    "dga_only": dict(ipl_phase=None, base_only=True, modules={"dga": dict(DGA_DEFAULTS)}),

    # MERGETUNE without IPL: valid ONLY on top of a fine-tuned base -> use Phase-A CoOp
    # as the fine-tuned endpoint. (On pure zero-shot there is nothing to merge, so
    # a "mergetune on zero-shot" experiment is intentionally omitted.)
    "mergetune_coop": dict(ipl_phase="A", modules={"mergetune": dict(MERGETUNE_DEFAULTS)}),
}


def get_experiment(name: str) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. Available:\n  " +
                       "\n  ".join(EXPERIMENTS))
    return EXPERIMENTS[name]
