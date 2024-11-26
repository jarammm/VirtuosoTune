import torch
import torch.nn as nn
from model_utils import make_higher_node, span_measure_to_note_num
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence



class MeasureGRU(nn.Module):
  """
  A GRU-based model for processing sequences at the measure level, 
  incorporating contextual attention for feature extraction.

  Args:
      input_size (int): Size of the input feature vectors.
      hidden_size (int): Size of the GRU hidden state.
      num_layers (int): Number of GRU layers.
      num_head (int): Number of attention heads.
      dropout (float): Dropout rate for the GRU.

  Attributes:
      attention (ContextAttention): Attention mechanism for extracting measure-level input.
      measure_rnn (nn.GRU): GRU for processing measure-level nodes.

  Methods:
      forward(x, measure_numbers):
          Processes a batch of sequences at the measure level using attention and GRU.
      one_step(x, hidden):
          Performs a single forward step for inference.
  """
  def __init__(self, input_size, hidden_size=128, num_layers=2, num_head=8, dropout=0.1):
    super().__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.attention = ContextAttention(input_size, num_head=num_head)
    self.measure_rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

  def forward(self, x, measure_numbers):
    """
      Forward pass for processing sequences at the measure level.

      Args:
          x (PackedSequence): Note-level GRU's output(=note hidden) of size `[total_seq_len, embedding_size]`, where:
              - `total_seq_len` is the total length of all sequences in the batch (sum of sequence lengths).
              - `embedding_size` is the size of each token's hidden vector.
          measure_numbers (PackedSequence): Measure number information as a size of [total_seq_len]

      Returns:
          packed_out (PackedSequence): Processed sequence at the note level, after measure-level aggregation.
      """
    if isinstance(x, PackedSequence):
      padded_x, batch_lens = pad_packed_sequence(x, batch_first=True)
      padded_measure_numbers, _ = pad_packed_sequence(measure_numbers, batch_first=True)
      measure_nodes = make_higher_node(padded_x, self.attention, padded_measure_numbers)
      out, _ = self.measure_rnn(measure_nodes)
      span_out = span_measure_to_note_num(out, padded_measure_numbers)
      packed_out =  pack_padded_sequence(span_out, batch_lens, batch_first=True, enforce_sorted=False)
      assert (packed_out.sorted_indices == x.sorted_indices).all()
      return packed_out
    else:
      measure_nodes = make_higher_node(x, self.attention, measure_numbers)
      out, _ = self.measure_rnn(measure_nodes)
      span_out = span_measure_to_note_num(out, measure_numbers)
      return span_out

  def one_step(self, x, hidden):
    node = self.attention(x)
    return self.measure_rnn(node.unsqueeze(1), hidden)


class ContextAttention(nn.Module):
    """
    A multi-head attention mechanism for contextual feature extraction.

    Args:
        size (int): Size of the attention_net.
        num_head (int): Number of attention heads.

    Attributes:
        attention_net (nn.Linear): Linear layer for projecting input features.
        num_head (int): Number of attention heads.
        head_size (int): Size of each attention head.
        context_vector (torch.nn.Parameter): Context vector for computing attention weights.

    Methods:
        get_attention(x):
            Computes attention weights for the input sequence.
        forward(x):
            Applies the attention mechanism to the input sequence and computes the weighted sum.
    """
    
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size/num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
            similarity[x.sum(-1)==0] = -1e10 # mask out zero padded_ones
            softmax_weight = torch.softmax(similarity, dim=1)

            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention


class DurPitchDecoder(nn.Module):
  def __init__(self, net_param, pitch_vocab_size, dur_vocab_size) -> None:
    super().__init__()

    self.pitch_proj = nn.Linear(net_param.dec.input_size, pitch_vocab_size)
    self.dur_proj = nn.Sequential(
      nn.Linear(net_param.dec.input_size + net_param.emb.main, net_param.dec.hidden_size),
      nn.ReLU(),
      nn.Dropout(net_param.dec.dropout),
      nn.Linear(net_param.dec.hidden_size, net_param.dec.hidden_size),
      nn.ReLU(),
      nn.Dropout(net_param.dec.dropout),
      nn.Linear(net_param.dec.hidden_size, dur_vocab_size)
    )

  def forward(self, x, pitch_emb=None, pitch_range=[]):
    pitch_logit = self.pitch_proj(x)
    if isinstance(pitch_emb, torch.nn.Module):
      pitch_sample = torch.multinomial(torch.softmax(pitch_logit[0], dim=-1), 1)
      pitch_emb_value = pitch_emb(pitch_sample)
      dur_logit = self.dur_proj(torch.cat([x, pitch_emb_value], dim=-1))
      if pitch_sample in pitch_range:
        dur_logit[..., :2] = -1e9
      dur_sample = torch.multinomial(torch.softmax(dur_logit[0], dim=-1), 1)
      return pitch_sample.squeeze(), dur_sample.squeeze()
    else: # if pitch_emb is a tensor
      dur_logit = self.dur_proj(torch.cat([x, pitch_emb], dim=-1))
      return torch.cat([pitch_logit, dur_logit], dim=-1)