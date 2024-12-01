import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from module import MeasureGRU, DurPitchDecoder
from omegaconf import DictConfig



class LanguageModel(nn.Module):
  def __init__(self, vocab_size: dict, net_param):
    super().__init__()
    self.net_param = net_param
    self.vocab_size = [x for x in vocab_size.values()]
    self.vocab_size_dict = vocab_size
    self.hidden_size = net_param.note.hidden_size
    self._make_embedding_layer()
    self.rnn = nn.GRU(self.hidden_size,
                      self.hidden_size,
                      num_layers=net_param.note.num_layers,
                      batch_first=True)
    self._make_projection_layer()

  @property
  def device(self):
      return next(self.parameters()).device

  def _make_embedding_layer(self):
    self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.net_param.emb.emb_size)

  def _make_projection_layer(self):
    self.proj = nn.Linear(self.hidden_size, self.vocab_size)
  
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      return emb
    else:
      emb = self.emb(input_seq)
      return emb

  def _apply_softmax(self, logit):
    return logit.softmax(dim=-1)


  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      logit = self.proj(hidden.data) # output: [num_total_notes x vocab_size].
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])

    else:
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      logit = self.proj(hidden)
      prob = self._apply_softmax(logit)

    return prob

  def _prepare_start_token(self, start_token_idx):
    return torch.LongTensor([start_token_idx]).to(self.device)

  def _prepare_inference(self, start_token_idx, manual_seed):
    start_token = self._prepare_start_token(start_token_idx)
    last_hidden = torch.zeros([self.rnn.num_layers, 1, self.rnn.hidden_size]).to(self.device)
    torch.manual_seed(manual_seed)
    return start_token, last_hidden


  def inference(self, start_token_idx=1, manual_seed=0):
    '''
    x can be just start token or length of T
    '''
    with torch.inference_mode():
      curr_token, last_hidden = self._prepare_inference(start_token_idx, manual_seed)
      total_out = []
      while True:
        emb = self.emb(curr_token.unsqueeze(0)) # embedding vector 변환 [1,128] -> [1, 1, 128]
        hidden, last_hidden = self.rnn(emb, last_hidden)
        logit = self.proj(hidden)
        prob = torch.softmax(logit, dim=-1)
        curr_token = prob.squeeze().multinomial(num_samples=1)
        if curr_token == 2: # Generated End token
          break 
        total_out.append(curr_token)
      return torch.cat(total_out, dim=0)


class PitchDurModel(LanguageModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)
    self.rnn = nn.GRU(self.net_param.emb.total_size,
                      self.hidden_size,
                      num_layers=net_param.note.num_layers,
                      batch_first=True)

  def _make_embedding_layer(self):
    self.emb = MultiEmbedding(self.vocab_size_dict, self.net_param)

  def _make_projection_layer(self):
    self.proj = nn.Linear(self.hidden_size, self.vocab_size[0] + self.vocab_size[1])

  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
    else:
      assert input_seq.ndim == 3
      emb = self.emb(input_seq)
    return emb
  
  def _apply_softmax(self, logit):
    prob = logit[:, :self.vocab_size[0]].softmax(dim=-1)
    prob = torch.cat([prob, logit[:, self.vocab_size[0]:].softmax(dim=-1)], dim=1)
    return prob

  def _sample_by_token_type(self, prob, vocab):
    """
    Sample indices of token type for def '_inference_one_step'
    Args:
        prob (torch.Tensor): 1D tensor [self.vocab_size[0]+self.vocab_size[1]]
            - self.vocab_size[0]: size of main values
            - self.vocab_size[1]: size of dur values
        vocab (NoteMusicTokenVocab)

    Returns:
       torch.Tensor: tensor([[pitch_idx, dur_idx, pitch_class_idx, octave_idx]])
          - If it's not a pitch, then pitch_class_idx, octave_idx are 0.
    """
    main_prob = prob[:self.vocab_size[0]]
    dur_prob = prob[self.vocab_size[0]:]
    main_token = main_prob.multinomial(num_samples=1)
    if 'pitch' in vocab.vocab['main'][main_token]:
      dur_prob[:3] = 0
      dur_token = dur_prob.multinomial(num_samples=1)
    else:
      dur_token = torch.tensor([0]).to(main_prob.device)
    
    converted_out = vocab.convert_inference_token(main_token, dur_token)
    return torch.tensor(converted_out, dtype=torch.long).to(main_prob.device).unsqueeze(0)

  def _prepare_start_token(self, start_token_idx):
    return torch.LongTensor([[start_token_idx, start_token_idx]]).to(self.device)

  def prepare_global_info(self, vocab, header):
    """
    Returns:
        header (dict): Default header from vocab.get_default_header() is
            {'key':'C Major', 'meter':'4/4', 'unit note length':'1/8', 'rhythm':'reel'}
        header_idx (list): header_idx list consist of
            [key, meter, unit_length, rhythm, root, mode, key_sig, numer, denom, is_compound, is_triple]
    """
    if header is None:
      header = vocab.get_default_header()
    header_idx = vocab.encode_header(header)

    return header, torch.LongTensor([header_idx]).to(self.device)

  def inference(self, vocab, manual_seed=0, header=None):
    header, global_condition = self.prepare_global_info(vocab, header)
    start_token_idx = vocab.vocab['main'].index('<start>')
    total_out = []
    curr_token, last_hidden = self._prepare_inference(start_token_idx, manual_seed)
    while True:
      curr_token = torch.cat([curr_token, global_condition], dim=-1)
      emb = self._get_embedding(curr_token.unsqueeze(0)) # embedding vector 변환 [1,128] -> [1, 1, 128]
      hidden, last_hidden = self.rnn(emb, last_hidden)
      logit = self.proj(hidden)
      prob = self._apply_softmax(logit)
      curr_token = self._sample_by_token_type(prob.squeeze(), vocab)
      if 2 in curr_token: # Generated End token
        break 
      total_out.append(curr_token)
    return torch.cat(total_out, dim=0)

class MeasureInfoModel(PitchDurModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)
    self.rnn = nn.GRU(net_param.emb.total_size, 
                      net_param.note.hidden_size, 
                      num_layers=net_param.note.num_layers, 
                      dropout=net_param.note.dropout,
                      batch_first=True)

  def _make_embedding_layer(self):
    self.emb = MultiEmbedding(self.vocab_size_dict, self.net_param.emb)

  def _get_measure_info(self, measure_info, vocab):
    idx, offset = measure_info
    idx = 'm_idx:' + str(idx)
    offset = 'm_offset:' + str(offset)
    return torch.LongTensor([[vocab.tok2idx['m_idx'][idx], vocab.tok2idx['m_offset'][offset]]]).to(self.device)

  def _prepare_start_token(self, vocab):
    # <start>, <pad>, <m_idx:0>, <m_offset:0>
    out = vocab.prepare_start_token()
    return torch.LongTensor([out]).to(self.device)

  def _inference_one_step(self, *args, **kwargs):
    curr_token, last_hidden, vocab = args
    emb = self._get_embedding(curr_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    logit = self.proj(hidden)
    prob = self._apply_softmax(logit)
    curr_token = self._sample_by_token_type(prob.squeeze(), vocab)

    return curr_token, last_hidden


  def inference(self, vocab, manual_seed=0, header=None):
    with torch.inference_mode():
      header, global_condition = self.prepare_global_info(vocab, header)
      measure_sampler = MeasureSampler(vocab, header)
      start_token, last_hidden, total_out = self._prepare_inference(vocab, header, manual_seed)
      curr_token = torch.cat([start_token, global_condition], dim=-1)

      while True:
        curr_token, last_hidden = self._inference_one_step(curr_token, last_hidden, vocab)
        if 2 in curr_token: # Generated End token
          break 
        total_out.append(curr_token)

        measure_sampler.update(curr_token) # update measure info
        measure_token = measure_sampler.get_measure_info_tensor().to(self.device)
        curr_token = torch.cat([curr_token, measure_token, global_condition], dim=-1)

    return torch.cat(total_out, dim=0)


class MultiEmbedding(nn.Module):
  """
  A multi-input embedding layer that combines embeddings from multiple vocabularies.
  Each vocabulary has its own embedding layer, and their outputs are concatenated along the last dimension.

  Args:
      vocab_sizes (Dict): A dictionary where keys are vocabulary names and values are the sizes of the vocabularies.
      vocab_param (DictConfig): A configuration object or dictionary containing the embedding sizes for each vocabulary key in `vocab_sizes`.

  Attributes:
      layers (nn.ModuleList): A list of embedding layers, one for each vocabulary, where the size is determined by `vocab_param`.

  Methods:
      forward(x): Combines embeddings for each input token and concatenates them.
      get_embedding_size(vocab_sizes, vocab_param): Determines the embedding sizes for each vocabulary.
  """
    
  def __init__(self, vocab_sizes: dict, vocab_param) -> None:
    super().__init__()
    self.layers = []
    embedding_sizes = self.get_embedding_size(vocab_sizes, vocab_param)
    for vocab_size, embedding_size in zip(vocab_sizes.values(), embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    """
    Forward pass for the MultiEmbedding layer.

    Args:
        x (torch.Tensor): Input tensor of shape `[total_seq_len, num_vocab]`.
            Each `[..., i]` slice corresponds to a token index for the `i`-th vocabulary.

    Returns:
        torch.Tensor: Concatenated embeddings of shape `[total_seq_len, total_embedding_size]`,
        where `total_embedding_size` is the sum of all individual embedding sizes.
    """
    return torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)

  def get_embedding_size(self, vocab_sizes, vocab_param):
    embedding_sizes = [getattr(vocab_param, vocab_key) for vocab_key in vocab_sizes.keys()]
    return embedding_sizes


class MeasureHierarchyModel(MeasureInfoModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)
    self.measure_rnn = MeasureGRU(net_param.note.hidden_size, 
                                  net_param.measure.hidden_size,
                                  num_layers=net_param.measure.num_layers, 
                                  dropout=net_param.measure.dropout)

  def _make_projection_layer(self):
    self.proj = nn.Linear((self.net_param.note.hidden_size + self.net_param.measure.hidden_size), self.vocab_size[0] + self.vocab_size[1])
  

  def forward(self, input_seq, measure_numbers):
    '''
    token -> rnn note_embedding                 -> projection -> softmax -> prob(pitch, duration)
                  | context attention          ^
                   -> rnn measure_embedding   _|(cat)
    '''
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      note_hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(note_hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([note_hidden.data, measure_hidden.data], dim=-1), note_hidden.batch_sizes, note_hidden.sorted_indices, note_hidden.unsorted_indices)
      logit = self.proj(cat_hidden.data)
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self._get_embedding(input_seq)
      note_hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(note_hidden, measure_numbers)
      
      cat_hidden = torch.cat([note_hidden, measure_hidden], dim=-1)
      logit = self.proj(cat_hidden)
      prob = self._apply_softmax(logit)

    return prob

  def _inference_one_step(self, *args, **kwargs):
    curr_token, last_hidden, last_measure_out, vocab = args
    emb = self._get_embedding(curr_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    cat_hidden = torch.cat([hidden, last_measure_out], dim=-1)
    logit = self.proj(cat_hidden)
    prob = self._apply_softmax(logit)
    curr_token = self._sample_by_token_type(prob.squeeze(), vocab)

    return curr_token, last_hidden

  def _prepare_start_token(self, vocab, header):
    out = vocab.prepare_start_token(header)
    return torch.LongTensor([out]).to(self.device)
  
  def _prepare_inference(self, vocab, header, manual_seed):
    start_token = self._prepare_start_token(vocab, header) # 주어진 header에 따른 시작 토큰 생성
    last_hidden = torch.zeros([self.rnn.num_layers, 1, self.rnn.hidden_size]).to(self.device)
    last_measure_out = torch.zeros([1, 1, self.measure_rnn.hidden_size]).to(self.device)
    last_measure_hidden = torch.zeros([self.measure_rnn.num_layers, 1, self.measure_rnn.hidden_size]).to(self.device)
    torch.manual_seed(manual_seed)
    return start_token, last_hidden, last_measure_out, last_measure_hidden

  def inference(self, vocab, manual_seed=0, header=None):
    total_hidden, total_out = [], []
    with torch.inference_mode():
      header, global_condition = self.prepare_global_info(vocab, header)
      measure_sampler = MeasureSampler(vocab, header)

      start_token, last_hidden, last_measure_out, last_measure_hidden = self._prepare_inference(vocab, header, manual_seed)
      curr_token = torch.cat([start_token, global_condition], dim=-1)

      prev_measure_num = 0

      while True:
        curr_token, last_hidden = self._inference_one_step(curr_token, last_hidden, last_measure_out, vocab)
        total_hidden.append(last_hidden[-1])

        if curr_token[0,0] == 2: # Generated End token
          break 
        total_out.append(curr_token)

        measure_sampler.update(curr_token)
        if measure_sampler.measure_number != prev_measure_num:
          last_measure_out, last_measure_hidden = self.measure_rnn.one_step(torch.cat(total_hidden, dim=0).unsqueeze(0), last_measure_hidden)
          prev_measure_num = measure_sampler.measure_number
          total_hidden = []
        
        measure_token = measure_sampler.get_measure_info_tensor().to(self.device)
        curr_token = torch.cat([curr_token, measure_token, global_condition], dim=-1)

    return torch.cat(total_out, dim=0)


class MeasureNoteModel(MeasureHierarchyModel):
  """
  A hierarchical model for music sequence generation that incorporates 
  note-level, measure-level, and final-level information.

  Inherits from MeasureHierarchyModel, which uses note-level and measure-level 
  representations. MeasureNoteModel extends this functionality by adding 
  a final-level GRU layer to further combine note and measure-level features.

  Attributes:
      vocab_size (List): The vocabulary size for note and duration tokens.
      net_param (DictConfig): A DictConfig containing network parameters for embedding, RNNs, and other components.
      final_rnn (torch.nn.GRU): The final GRU layer combining note and measure hidden states.
      proj (torch.nn.Linear): A projection layer mapping the final hidden state to the vocabulary size.

  Methods:
      forward(input_seq, measure_numbers):
          Processes a sequence of inputs and produces probabilities for the next tokens.
      _make_projection_layer():
          Creates the projection layer for the model.
      _prepare_inference(vocab, header, manual_seed):
          Prepares the model's state for inference.
      _inference_one_step(*args, **kwargs):
          Performs a single step of inference.
      inference(vocab, manual_seed=0, header=None):
          Generates sequences token-by-token using the model.
  """
    
  def __init__(self, vocab_size: list, net_param: DictConfig):
    super().__init__(vocab_size, net_param)
    self.final_rnn = nn.GRU((self.net_param.note.hidden_size + self.net_param.measure.hidden_size),
                            self.net_param.final.hidden_size,
                            num_layers=self.net_param.final.num_layers,
                            dropout=self.net_param.final.dropout,
                            batch_first=True)
    self._make_projection_layer()
  
  def _make_projection_layer(self):
    self.proj = nn.Linear(self.net_param.final.hidden_size, self.vocab_size[0] + self.vocab_size[1])

  def forward(self, input_seq, measure_numbers):
    """
    Process:
        token -> rnn note_embedding                -> rnn final_embedding -> projection -> softmax -> prob(pitch, duration)
                      | context attention         ^
                      -> rnn measure_embedding   _|(cat)
    
    Args:
        input_seq (PackedSequence): Input sequence(melody). Has a shape of [total_seq_len, vocab_type_num].
        measure_numbers (PackedSequence): Measure number information per note. Has a shape of [total_seq_len].

    Returns:
        prob (PackedSequence): Output probabilities for each token in the sequence.
            Has a shape of [total_seq_len, number of main + dur classes].
    """
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      note_hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(note_hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([note_hidden.data, measure_hidden.data], dim=-1), note_hidden.batch_sizes, note_hidden.sorted_indices, note_hidden.unsorted_indices)
      final_hidden, _ = self.final_rnn(cat_hidden)

      logit = self.proj(final_hidden.data)
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self._get_embedding(input_seq)
      note_hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(note_hidden, measure_numbers)
      
      cat_hidden = torch.cat([note_hidden, measure_hidden], dim=-1)
      final_hidden, _ = self.final_rnn(cat_hidden)
      logit = self.proj(final_hidden)
      prob = self._apply_softmax(logit)
      
    return prob

  def _prepare_inference(self, vocab, header, manual_seed):
    start_token, last_hidden, last_measure_out, last_measure_hidden = super()._prepare_inference(vocab, header, manual_seed)
    last_final_hidden = torch.zeros([self.final_rnn.num_layers, 1, self.final_rnn.hidden_size]).to(self.device)
    return start_token, last_hidden, last_measure_out, last_measure_hidden, last_final_hidden
  
  def _inference_one_step(self, *args, **kwargs):
    curr_token, last_hidden, last_measure_out, last_final_hidden, vocab = args
    emb = self._get_embedding(curr_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden) # emb : [batch_size,seq_len,embedding_size] / last_hidden : [rnn_num_layers, batch_size, hidden_size]
    cat_hidden = torch.cat([hidden, last_measure_out], dim=-1)
    final_hidden, last_final_hidden = self.final_rnn(cat_hidden, last_final_hidden)
    logit = self.proj(final_hidden)
    prob = self._apply_softmax(logit)
    curr_token = self._sample_by_token_type(prob.squeeze(), vocab)
    return curr_token, last_hidden, last_final_hidden

  def inference(self, vocab, manual_seed=0, header=None):
    """
    This method runs the inference process, sampling notes and measures step-by-step using the GRU-based model. 
    It integrates global conditions, measure information, and note information from previous steps to iteratively 
    produce the final output sequence.

    Args:
        vocab (dict): Vocabulary containing the mapping of tokens to their corresponding indices.
        manual_seed (int, optional): Random seed for controlling the reproducibility of the sampling process. Default is 0.
        header (dict, optional): Dictionary containing global metadata such as rhythm, key, and etc.

    Returns:
        torch.Tensor: The concatenated tensor of generated note tokens.

    Notes:
        - The inference process maintains several key components throughout:
            - `last_measure_out`: Initially set to `torch.zeros` until all notes for the first measure are generated.
            - `last_measure_hidden`: Not updated until all notes for the current measure are generated.
            - `start_token`: A concatenated tensor of note-specific components:
                - `start_token[:4]`: [[pitch_idx, dur_idx, pitch_class_idx, octave_idx]]
                - `start_token[4:9]`: [[m_idx, m_idx_mod4, m_offset, is_onbeat, is_middle_beat]]
            - `global_condition`: Metadata that includes [key, meter, unit_length, rhythm, root, mode, key_sig, numer, denom, is_compound, is_triple].
            - `curr_token`: Tensor shape of [1,20], contains musical information of a note.
  """
    total_hidden, total_out = [], []
    with torch.inference_mode():
      header, global_condition = self.prepare_global_info(vocab, header)
      measure_sampler = MeasureSampler(vocab, header)
      start_token, last_note_hidden, last_measure_out, last_measure_hidden, last_final_hidden = self._prepare_inference(vocab, header, manual_seed)
      curr_token = torch.cat([start_token, global_condition], dim=-1)

      prev_measure_num = 0

      while True:
        note_token, last_note_hidden, last_final_hidden = self._inference_one_step(curr_token, last_note_hidden, last_measure_out, last_final_hidden, vocab)
        total_hidden.append(last_note_hidden[-1])

        if note_token[0,0] == 2: # Generated End token
          break
        total_out.append(note_token)

        measure_sampler.update(note_token)
        if header['rhythm'] == 'reel':
          if measure_sampler.cur_m_offset == 4.0:
            total_out.append(torch.tensor([[0, 0, 0, 0]]).to(self.device))
        if measure_sampler.measure_number != prev_measure_num:
          last_measure_out, last_measure_hidden = self.measure_rnn.one_step(torch.cat(total_hidden, dim=0).unsqueeze(0), last_measure_hidden)
          prev_measure_num = measure_sampler.measure_number
          total_hidden = []
        
        measure_token = measure_sampler.get_measure_info_tensor().to(self.device) # KeyError: 'm_offset:9.75'
        curr_token = torch.cat([note_token, measure_token, global_condition], dim=-1)

    return torch.cat(total_out, dim=0)
  
  def generate_header(self, key_mode, header, manual_seed):
      if key_mode == 'random':
          if manual_seed % 10 < 7:
            header['key'] = 'C Major'
          elif manual_seed % 3 == 0:
            header['key'] = 'C minor'
          elif manual_seed % 3 < 1:
            header['key'] = 'C Dorian'
          else:
            header['key'] = 'C Mixolydian'
      return header
  
  def outer_inference(self, vocab, key_mode, header, manual_seed):
    header = self.generate_header(key_mode, header, manual_seed)
    header, global_condition = self.prepare_global_info(vocab, header)
    start_token, last_note_hidden, last_measure_out, last_measure_hidden, last_final_hidden = self._prepare_inference(vocab, header, manual_seed)
    curr_token = torch.cat([start_token, global_condition], dim=-1)
    return header, global_condition, curr_token, last_note_hidden, last_measure_out, last_measure_hidden, last_final_hidden
  
  def inner_inference(self, vocab, header, global_condition, curr_token, last_note_hidden, last_measure_out, last_measure_hidden, last_final_hidden):
    self.measure_sampler = MeasureSampler(vocab, header)
    prev_measure_num = 0
    total_hidden, total_out = [], []
    while len(total_out) < 3:
      note_token, last_note_hidden, last_final_hidden = self._inference_one_step(curr_token, last_note_hidden, last_measure_out, last_final_hidden, vocab)
      total_hidden.append(last_note_hidden[-1])

      if note_token[0,0] == 2: # Generated End token
        break
      total_out.append(note_token)

      self.measure_sampler.update(note_token)
      if header['rhythm'] == 'reel':
        if self.measure_sampler.cur_m_offset == 4.0:
          total_out.append(torch.tensor([[0, 0, 0, 0]]).to(self.device))
      if self.measure_sampler.measure_number != prev_measure_num:
        last_measure_out, last_measure_hidden = self.measure_rnn.one_step(torch.cat(total_hidden, dim=0).unsqueeze(0), last_measure_hidden)
        prev_measure_num = self.measure_sampler.measure_number
        total_hidden = []
      
      measure_token = self.measure_sampler.get_measure_info_tensor().to(self.device)
      curr_token = torch.cat([note_token, measure_token, global_condition], dim=-1)

    return torch.cat(total_out, dim=0)
    
    
  def inference_onnx(self, vocab, key_mode, header, manual_seed=0):
    while True:
      try:
        header, global_condition, curr_token, last_note_hidden, last_measure_out, last_measure_hidden, last_final_hidden = self.outer_inference(vocab, key_mode, header, manual_seed.item())
        out = self.inner_inference(vocab, header, global_condition, curr_token, last_note_hidden, last_measure_out, last_measure_hidden, last_final_hidden)
        return out, manual_seed
      except Exception as e:
        print(f"decoding failed: {e}")
      manual_seed += 1


class MeasureNotePitchFirstModel(MeasureNoteModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)

  def _make_projection_layer(self):
    self.proj = DurPitchDecoder(self.net_param, self.vocab_size[0], self.vocab_size[1])

  def time_shifted_pitch_emb(self, emb):
    '''
    Get time-shifted pitch embedding to feed to final projection layer
    This model's projection layer first estimates pitch and then estimates duration

    '''
    end_tokens = torch.tensor([2],dtype=torch.long).to(emb.data.device) # Use 2 to represent end token
    end_vec = self.emb.layers[0](end_tokens) # pitch embedding vector 
    if isinstance(emb, PackedSequence):
      padded_emb, batch_lens = pad_packed_sequence(emb, batch_first=True)
      shifted_emb = torch.cat([padded_emb[:, 1:, :self.emb.layers[0].embedding_dim], end_vec.expand(padded_emb.shape[0], 1, -1)], dim=1)
      packed_emb = pack_padded_sequence(shifted_emb, batch_lens, batch_first=True, enforce_sorted=False)
      assert (packed_emb.sorted_indices == emb.sorted_indices).all()
      return packed_emb
    else:
      return torch.cat([emb[:, 1: :self.emb.layers[0].embedding_dim], end_vec.expand(emb.shape[0], 1, -1) ], dim=-1)

  def _inference_one_step(self, *args, **kwargs):
    curr_token, last_hidden, last_measure_out, last_final_hidden, vocab = args
    emb = self._get_embedding(curr_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    cat_hidden = torch.cat([hidden, last_measure_out], dim=-1)
    final_hidden, last_final_hidden = self.final_rnn(cat_hidden, last_final_hidden)
    main_token, dur_token = self.proj(final_hidden, self.emb.layers[0], vocab.pitch_range)

    converted_out = vocab.convert_inference_token(main_token, dur_token)
    curr_token = torch.tensor(converted_out, dtype=torch.long).to(emb.device).unsqueeze(0)
    return curr_token, last_hidden, last_final_hidden

  def forward(self, input_seq, measure_numbers):
    '''
    token -> rnn note_embedding                 -> projection -> pitch, duration
                  | context attention          ^
                   -> rnn measure_embedding   _|(cat)
    '''
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([hidden.data, measure_hidden.data], dim=-1), hidden.batch_sizes, hidden.sorted_indices, hidden.unsorted_indices)
      final_hidden, _ = self.final_rnn(cat_hidden)

      pitch_vec = self.time_shifted_pitch_emb(emb)
      logit = self.proj(final_hidden.data, pitch_vec.data) # output: [num_total_notes x vocab_size].
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)
      
      cat_hidden = torch.cat([hidden, measure_hidden], dim=-1)
      final_hidden, _ = self.final_rnn(cat_hidden)
      pitch_vec = self.time_shifted_pitch_emb(emb)
      logit = self.proj(final_hidden, pitch_vec)
      prob = self._apply_softmax(logit)

    return prob


class MeasureSampler:
  def __init__(self, vocab, header):
    self.vocab = vocab
    self.header = header

    self.cur_m_offset = 0
    self.cur_m_index = 0

    self.tuplet_count = 0
    self.tuplet_duration = 0
    self.full_measure_duration = 8 # TODO: 박자별로 조절
    self.first_ending_offset = 0
    self.measure_number = 0
  
  def get_measure_info_tensor(self):
    idx, offset = self.cur_m_index, self.cur_m_offset
    idx = 'm_idx:' + str(idx)
    offset = 'm_offset:' + str(float(offset))

    return torch.tensor([self.vocab.encode_m_idx(idx) + self.vocab.encode_m_offset(offset, self.header)], dtype=torch.long)

  def update(self, curr_token):
    sampled_token_str = self.vocab.vocab['main'][curr_token[0,0].item()]
    if '|' in sampled_token_str:
      if self.cur_m_offset > self.full_measure_duration / 2:
        self.cur_m_index += 1
        self.measure_number += 1
      self.cur_m_offset = 0
      if '|1' in sampled_token_str:
        self.first_ending_offset = self.cur_m_index
      if '|2' in sampled_token_str:
        self.cur_m_index = self.first_ending_offset
    if '|:' in sampled_token_str:
      self.cur_m_index = 0
    elif '(3' in sampled_token_str: #TODO: Solve it with regex
      self.tuplet_count = int(sampled_token_str.replace('(', ''))
    elif 'pitch' in sampled_token_str:
      sampled_dur = float(self.vocab.vocab['dur'][curr_token[0,1].item()].replace('dur', ''))
      if self.tuplet_count == 0:
        if self.tuplet_duration:
          self.cur_m_offset += self.tuplet_duration * 2
          self.tuplet_duration = 0
        else:
          self.cur_m_offset += sampled_dur
      else:
        self.tuplet_count -= 1
        self.tuplet_duration = sampled_dur
        if self.tuplet_count == 0:
          self.cur_m_offset += self.tuplet_duration * 2
          self.tuplet_duration = 0
    else:
      self.tuplet_count = 0
      self.tuplet_duration = 0


class MeasureGPT(MeasureInfoModel):
  def __init__(self, vocab_size, hidden_size, dropout=0.1):
    super().__init__(vocab_size, hidden_size, dropout)


  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      logit = self.proj(hidden.data) # output: [num_total_notes x vocab_size].
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      logit = self.proj(hidden)
      prob = self._apply_softmax(logit)

    return prob
