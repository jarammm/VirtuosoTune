import torch
import torch.nn as nn
from pathlib import Path
import model_zoo
import data_utils
import vocab_utils
import argparse

class MeasureNoteWrapper(nn.Module):
    def __init__(self, model, vocab, header, key_mode):
        super(MeasureNoteWrapper, self).__init__()
        self.model = model
        self.vocab = vocab
        self.header = header
        self.key_mode = key_mode
    def forward(self, seed):
        return self.model.inference_onnx(self.vocab, self.key_mode, self.header, seed)


def export_to_onnx(args, model, vocab, header):
    inference_model = MeasureNoteWrapper(model, vocab, header, args.key_mode)
    dummy_manual_seed = torch.tensor(args.seed, dtype=torch.int32).to(args.device)
    
    torch.onnx.export(
        inference_model,
        (dummy_manual_seed),  # Dummy inputs
        args.onnx_save_path,  # Path to save the ONNX file
        input_names=["manual_seed"],  # Names for input tensors
        output_names=["generated_notes", "manual_seed"],  # Name for output tensor
        opset_version=17  # ONNX opset version
    )
    print(f"Model exported to {args.onnx_save_path}")

def prepare_model_for_export(args, pretrained=False):
    header =  {'key':f'C {args.key_mode}', 'meter':'4/4', 'unit note length':'1/8', 'rhythm':args.rhythm}
    yaml_path = list(args.experiment_path.glob('*.yaml'))[0]
    vocab_path = list(args.experiment_path.glob('*vocab.json'))[0]
    checkpoint_path = None
    if pretrained:
      checkpoint_list = list(args.experiment_path.glob('*.pt'))
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
    model.to(args.device)

    return model, vocab, header

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_samples', type=int, default=10)
  parser.add_argument('--experiment_path', type=Path, default=Path("experiments/20241126-163657-MeasureNoteModel_512_32_100000_0.0003"))
  parser.add_argument('--onnx_save_path', type=str, default="MeasureNoteModel.onnx")
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--key_mode', type=str, default='random', choices=['random', 'Major', 'minor', 'Dorian', 'Mixolydian'])
  parser.add_argument('--rhythm', type=str, default='reel', choices=['reel', 'jig'])
  parser.add_argument('--seed', type=int, default=4035) # 4035 was the seed for the first-prize winning tune

  return parser


if __name__ == "__main__":
  args = get_parser().parse_args()
  model, vocab, header = prepare_model_for_export(args)
  export_to_onnx(args, model, vocab, header)