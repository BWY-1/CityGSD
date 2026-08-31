import torch
import os

def load_mlp_checkpoint(model_dir:str, iter_num:int=-1):
    """
    加载citygs‑x保存的checkpoint
    iter_num=-1 读取最新迭代
    """
    ckpt_path = None
    if iter_num == -1:
        # 自动找最新 .pth
        files = [f for f in os.listdir(model_dir) if f.endswith(".pth") and "ckpt" in f]
        files.sort()
        ckpt_path = os.path.join(model_dir, files[-1])
    else:
        ckpt_path = os.path.join(model_dir, f"ckpt_{iter_num}.pth")
    print(f"load checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["gaussians"].state_dict()
    # 过滤：只保留 shared_mlp 相关权重key
    mlp_keys = [k for k in state_dict.keys() if "shared_mlp" in k]
    mlp_state = {k: state_dict[k] for k in mlp_keys}
    return mlp_state, mlp_keys


if __name__ == "__main__":
    # ========= 修改这里路径 =========
    source_model_path = "/mnt/4TB/CameraA/cameraA_4080_R4"    # 原始预训练模型
    updated_model_path = "/mnt/4TB/CameraA/cameraA_update_small_roi" # update输出目录
    source_iter = -1
    updated_iter = 100
    # =================================

    source_mlp, src_keys = load_mlp_checkpoint(source_model_path, source_iter)
    updated_mlp, upd_keys = load_mlp_checkpoint(updated_model_path, updated_iter)

    # key集合必须完全一致
    assert set(src_keys) == set(upd_keys), "shared_mlp key列表不一致！"

    all_ok = True
    for key in src_keys:
        t1 = source_mlp[key]
        t2 = updated_mlp[key]
        equal_flag = torch.equal(t1, t2)
        max_diff = torch.max(torch.abs(t1 - t2)).item()
        print(f"[{key}] equal={equal_flag:.0f}, max_abs_diff={max_diff:.2e}")
        if not equal_flag:
            all_ok = False

    print("\n===== 验证结果 =====")
    if all_ok:
        print("✅ Shared‑MLP Invariance PASS：所有权重完全未改动，freeze生效")
    else:
        print("❌ FAIL：部分shared‑mlp张量发生变化！--freeze_shared_mlp没有生效")
