"""重新导出 ONNX — 使用匹诺曹的 ACTExportWrapper (infer_cvae=False)"""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

MODEL_PATH = PROJECT_ROOT / "output/train/model.pt"
ONNX_PATH = PROJECT_ROOT / "output/train/model.onnx"


class ACTExportWrapper(torch.nn.Module):
    """匹诺曹使用的包装器：infer_cvae=False 跳过 CVAE 采样"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, state):
        output = self.model(image, state, action_target=None, infer_cvae=False)
        return output["action"]


def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    config_dict = ckpt.get("config", {})
    config = ACTConfig(**config_dict)
    model = ACTModel(config)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ACTExportWrapper(model)
    wrapper.eval()

    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_state = torch.zeros(1, config_dict.get("state_dim", 2))

    # 先跑一次 PyTorch 验证
    with torch.no_grad():
        pt_out = wrapper(dummy_image, dummy_state)
    print(f"PyTorch 测试输出: {pt_out.shape}")

    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_state),
        ONNX_PATH,
        input_names=["images", "state"],
        output_names=["action"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"ONNX 导出: {ONNX_PATH}")

    # 验证 ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(str(ONNX_PATH))
    onnx_out = session.run(None, {"images": dummy_image.numpy(), "state": dummy_state.numpy()})[0]
    diff = abs(pt_out.numpy() - onnx_out).max()
    print(f"PyTorch vs ONNX 最大误差: {diff:.6f}")
    print(f"模型大小: {ONNX_PATH.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
