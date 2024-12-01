import torch
import torch.nn as nn
from pathlib import Path
import model_zoo
import data_utils
import vocab_utils

class InferenceWrapper(nn.Module):
    """
    A wrapper to adapt the `inference` method of the model for ONNX export.
    """
    def __init__(self, model, vocab):
        super(InferenceWrapper, self).__init__()
        self.model = model
        self.vocab = vocab
        self.header = {
            "key": "C Major",
            "meter": "4/4",
            "unit note length": "1/8",
            "rhythm": "reel"
        }
    def forward(self, seed):
        """
        Wrap the inference function for ONNX export.
        """
        return self.model.inference(self.vocab, seed.item(), self.header)


def export_to_onnx(model, vocab, save_path, device="cpu"):
    """
    Export the MeasureInfoModel to ONNX format using its inference method.
    """
    # Wrap the model for ONNX export
    inference_model = InferenceWrapper(model, vocab)

    # Dummy inputs for ONNX export
    dummy_manual_seed = torch.tensor(1234, dtype=torch.int32).to(device)

    # Export the model to ONNX
    torch.onnx.export(
        inference_model,
        (dummy_manual_seed),  # Dummy inputs
        save_path,  # Path to save the ONNX file
        input_names=["manual_seed"],  # Names for input tensors
        output_names=["generated_notes"],  # Name for output tensor
        opset_version=17  # ONNX opset version
    )
    print(f"Model exported to {save_path}")

# 모델 준비 함수
def prepare_model_for_export(path: Path, device: str, pretrained=False):
    """
    Load the MeasureInfoModel, vocabulary, and configuration from the given path.
    """
    yaml_path = list(path.glob('*.yaml'))[0]
    vocab_path = list(path.glob('*vocab.json'))[0]
    checkpoint_path = None
    if pretrained:
      checkpoint_list = list(path.glob('*.pt'))
      checkpoint_list.sort(key=lambda x: int(x.stem.split('_')[-2].replace('iter', '')))
      checkpoint_path = checkpoint_list[-1]

    config = data_utils.read_yaml(yaml_path)
    model_name = config.nn_params.model_name
    vocab_name = config.nn_params.vocab_name
    net_param = config.nn_params

    vocab = getattr(vocab_utils, vocab_name)(json_path=vocab_path)
    config = data_utils.get_emb_total_size(config, vocab)
    model = getattr(model_zoo, model_name)(vocab.get_size(), net_param)
    
    if checkpoint_path:
      checkpoint = torch.load(checkpoint_path, map_location="cpu")
      model.load_state_dict(checkpoint["model"])

    model.eval()
    model.to(device)

    return model, vocab


# 경로 설정 및 모델 변환 실행
experiment_path = Path("experiments/20241126-163657-MeasureNoteModel_512_32_100000_0.0003")
onnx_save_path = "MeasureNoteModel.onnx"
device = "cpu"

# 모델 준비
model, vocab = prepare_model_for_export(experiment_path, device)

# ONNX로 변환
export_to_onnx(model, vocab, onnx_save_path, device)