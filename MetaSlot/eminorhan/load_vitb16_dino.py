import re
import torch as pt

def load_backbone_into_model(model,ckpt):
    # import pdb; pdb.set_trace()
    if isinstance(ckpt, dict):
        if "student" in ckpt and isinstance(ckpt["student"], dict):
            src = ckpt["student"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            src = ckpt["state_dict"]
        else:
            src = ckpt
    else:
        src = ckpt
    # src = ckpt["student"]

    # remove module. if present
    src = { (k[7:] if k.startswith("module.") else k): v for k, v in src.items() }

    # backbone.* | model.m.encode_backbone.1.*
    def map_key(k: str) -> str | None:
        if not k.startswith("backbone."):
            return None
        # backbone. -> model.m.encode_backbone[1].
        nk = re.sub(r"^backbone\.", "m.encode_backbone.1.", k)

        return nk

    mapped = {}
    for k, v in src.items():
        nk = map_key(k)
        if nk is not None:
            mapped[nk] = v

    
    dst_sd = model.state_dict()
    loadable = {k: v for k, v in mapped.items()
                if (k in dst_sd) and (dst_sd[k].shape == v.shape)}

    # 记录无法直接加载（缺失/形状不符）的键，便于你检查
    # missing_in_model = sorted(set(mapped) - set(dst_sd))
    shape_mismatch = sorted([k for k in mapped if k in dst_sd and dst_sd[k].shape != mapped[k].shape])

    print(f"[Info] 可加载参数数: {len(loadable)} / 映射后总数: {len(mapped)}")
    # if missing_in_model:
    #     print("[Skip] 目标模型不存在这些键（例如目标没有 cls_token 就会在此出现）:")
    #     for k in missing_in_model[:20]:
    #         print("  -", k)
    #     if len(missing_in_model) > 20:
    #         print(f"  ... 共 {len(missing_in_model)} 项")

    if shape_mismatch:
        print("[Skip] 形状不匹配（可能需要插值 pos_embed 或通道数不同）:")
        for k in shape_mismatch[:20]:
            print(f"  - {k}: ckpt {tuple(mapped[k].shape)} vs model {tuple(dst_sd[k].shape)}")
        if len(shape_mismatch) > 20:
            print(f"  ... 共 {len(shape_mismatch)} 项")

    # 加载（非严格，保留未匹配层为随机初始化）
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print("[Result] load_state_dict -> missing:", missing, " unexpected:", unexpected)

    return model

