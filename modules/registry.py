"""Registry: build the active ModuleStack from a flat experiment config."""
from __future__ import annotations

from .base import AdaptationModule, BuildContext, ModuleStack
from .sca_module import SCAModule
from .mergetune_module import MergeTuneModule
from .dga_module import DGAModule

REGISTRY = {
    "sca": SCAModule,
    "mergetune": MergeTuneModule,
    "dga": DGAModule,
}


def build_stack(exp_cfg: dict, clip_model, tokenizer, classnames,
                class_prompts, ipl_model, device) -> ModuleStack:
    """`exp_cfg['modules']` is a dict like {'sca': {...}, 'dga': {...}}.
    Only listed modules are built; IPL is always the base and lives outside."""
    modules: list[AdaptationModule] = []
    for name, mcfg in (exp_cfg.get("modules") or {}).items():
        if not mcfg or not mcfg.get("enabled", True):
            continue
        if name not in REGISTRY:
            raise KeyError(f"Unknown module '{name}'. Known: {list(REGISTRY)}")
        m = REGISTRY[name]()
        m.build(BuildContext(
            clip_model=clip_model, tokenizer=tokenizer, classnames=classnames,
            class_prompts=class_prompts, ipl_model=ipl_model, device=device, cfg=mcfg,
        ))
        modules.append(m.to(device))
    return ModuleStack(modules)
