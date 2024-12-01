import torch
import argparse
from pathlib import Path

import model_zoo
import data_utils
import vocab_utils
from decoding import LanguageModelDecoder
from tqdm.auto import tqdm
from pyabc import pyabc

import onnxruntime
import numpy as np

def inference(args):

  path = Path(args.path)
  if path.is_dir():
    yaml_path = list(path.glob('*.yaml'))[0]
    vocab_path = list(path.glob('*vocab.json'))[0]
    checkpoint_list = list(path.glob('*.onnx'))
    checkpoint_path = checkpoint_list[-1]

    config = data_utils.read_yaml(yaml_path)
    vocab_name = config.nn_params.vocab_name

    vocab = getattr(vocab_utils, vocab_name)(json_path= vocab_path)
    config = data_utils.get_emb_total_size(config, vocab)

    session = onnxruntime.InferenceSession(checkpoint_path)

  else:
    pass

  decoder =  LanguageModelDecoder(vocab, args.save_dir)
  args.save_dir.mkdir(parents=True, exist_ok=True)

  header =  {'key':f'C {args.key_mode}', 'meter':'4/4', 'unit note length':'1/8', 'rhythm':args.rhythm}
  num_generated = 0
  rand_seed = args.seed
  while num_generated < args.num_samples:
    if args.key_mode == 'random':
      if rand_seed % 10 < 7:
        header['key'] = 'C Major'
      elif rand_seed % 3 == 0:
        header['key'] = 'C minor'
      elif rand_seed % 3 < 1:
        header['key'] = 'C Dorian'
      else:
        header['key'] = 'C Mixolydian'

    try:
      inputs = {session.get_inputs()[0].name: np.array(rand_seed, dtype=np.int32)}
      out, rand_seed = session.run(None, inputs)
      out, rand_seed = torch.tensor(out), rand_seed.item()
      meta_string = f'X:1\nT:Title\nM:2/2\nL:{header["unit note length"]}\nK:{header["key"]}\n'
      gen_abc = decoder.decode(out, meta_string)     
      file_name = f'model_{yaml_path.stem}_seed_{rand_seed}'
      decoder(gen_abc, file_name, save_image=args.save_image, save_audio=args.save_audio, meta_string=meta_string)
      num_generated += 1
      print(f'generated {num_generated} tunes')
    except Exception as e:
      print(f"decoding failed: {e}")
    rand_seed += 1


def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=Path, default=Path('pre_trained/measure_note/'))
  parser.add_argument('--num_samples', type=int, default=1)
  parser.add_argument('--save_dir', type=Path, default=Path('generated'))
  parser.add_argument('--save_audio', action='store_true')
  parser.add_argument('--save_image', action='store_true')
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--key_mode', type=str, default='random', choices=['random', 'Major', 'minor', 'Dorian', 'Mixolydian'])
  parser.add_argument('--rhythm', type=str, default='reel', choices=['reel', 'jig'])
  parser.add_argument('--seed', type=int, default=4035) # 4035 was the seed for the first-prize winning tune

  return parser


if __name__ == "__main__":
  args = get_parser().parse_args()
  inference(args)