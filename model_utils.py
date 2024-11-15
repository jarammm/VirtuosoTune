import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



def find_boundaries(diff_boundary, higher_indices, i):
  '''
  diff_boundary (torch.Tensor): N x T
  measure_numbers (torch.Tensor): zero_padded N x T
  i (int): batch index
  '''
  out = [0] + (diff_boundary[diff_boundary[:,0]==i][:,1]+1 ).tolist() + [torch.max(torch.nonzero(higher_indices[i])).item()+1]
  if out[1] == 0: # if the first boundary occurs in 0, it will be duplicated
    out.pop(0)
  return out

def find_boundaries_batch(measure_numbers):
  '''
  find all boundaries and then make boundaries list per batch
  measure_numbers (torch.Tensor): zero_padded N x T
  '''
  diff_boundary = torch.nonzero(measure_numbers[:,1:] - measure_numbers[:,:-1]).cpu()
  return [find_boundaries(diff_boundary, measure_numbers, i) for i in range(len(measure_numbers))]


def get_softmax_by_boundary(similarity, boundaries, fn=torch.softmax):
  '''
  similarity = similarity of a single sequence of data (T x C)
  boundaries = list of a boundary index (T)
  '''
  return  [fn(similarity[boundaries[i-1]:boundaries[i],: ], dim=0)  \
              for i in range(1, len(boundaries))
                if boundaries[i-1] < boundaries[i] # sometimes, boundaries can start like [0, 0, ...]
          ]


def run_hierarchy_rnn_with_pack(sequence, rnn):
  '''
  sequence (torch.Tensor): zero-padded sequece of N x T x C
  lstm (torch.LSTM): LSTM layer
  '''
  batch_note_length = sequence.shape[1] - (sequence==0).all(dim=-1).sum(-1)
  packed_sequence = pack_padded_sequence(sequence, batch_note_length.cpu(), True, False )
  hidden_out, _ = rnn(packed_sequence)
  hidden_out, _ = pad_packed_sequence(hidden_out, True)

  return hidden_out


def make_higher_node(note_hidden, attention_weights, measure_numbers):
    similarity = attention_weights.get_attention(note_hidden)
    boundaries = find_boundaries_batch(measure_numbers)
    softmax_similarity = torch.nn.utils.rnn.pad_sequence(
      [torch.cat(get_softmax_by_boundary(similarity[batch_idx], boundaries[batch_idx]))
        for batch_idx in range(len(note_hidden))], 
      batch_first=True
    )
    
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.stack(note_hidden.split(split_size=attention_weights.head_size, dim=2), dim=2)
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
        weighted_x = weighted_x.view(x_split.shape[0], x_split.shape[1], note_hidden.shape[-1])
        higher_nodes = torch.nn.utils.rnn.pad_sequence([
          torch.cat([torch.sum(weighted_x[i:i+1,boundaries[i][j-1]:boundaries[i][j],: ], dim=1) for j in range(1, len(boundaries[i]))], dim=0) \
          for i in range(len(note_hidden))
        ], batch_first=True
        )
    else:
        weighted_sum = softmax_similarity * note_hidden
        higher_nodes = torch.cat([torch.sum(weighted_sum[:,boundaries[i-1]:boundaries[i],:], dim=1) 
                                for i in range(1, len(boundaries))]).unsqueeze(0)
    return higher_nodes


def span_measure_to_note_num(measure_out, measure_number):
  '''
  Broadcasting measure-level output into note-level,
  so that each note or token’s hidden output was concatenated with the measure-level output of the previous measure afterward.
  
  measure_out (torch.Tensor): N x T_measure x C
  measure_number (torch.Tensor): N x T_note x C
  '''
  zero_shifted_measure_number = measure_number - measure_number[:,0:1]
  len_note = cal_length_from_padded_measure_numbers(measure_number)

  batch_indices = torch.cat([torch.ones(length)*i for i, length in enumerate(len_note)]).long()
  note_indices = torch.cat([torch.arange(length) for length in len_note])
  measure_indices = torch.cat([zero_shifted_measure_number[i,:length] for i, length in enumerate(len_note)]).long()

  measure_indices = measure_indices - 1 # note has to get only previous measure info

  span_mat = torch.zeros(measure_number.shape[0], measure_number.shape[1], measure_out.shape[1]).to(measure_out.device)
  span_mat[batch_indices, note_indices, measure_indices] = 1
  span_mat[:, :, -1] = 0 # last measure is not used
  spanned_measure = torch.bmm(span_mat, measure_out)
  return spanned_measure

def cal_length_from_padded_measure_numbers(measure_numbers):
  '''
  measure_numbers (torch.Tensor): N x T, zero padded note_location_number

  output (torch.Tensor): N
  '''
  try:
    len_note = torch.min(torch.diff(measure_numbers,dim=1), dim=1)[1] + 1
  except:
    print("Error in cal_length_from_padded_measure_numbers:")
    print(measure_numbers)
    print(measure_numbers.shape)
    [print(measure_n) for measure_n in measure_numbers]
    print(torch.diff(measure_numbers,dim=1))
    print(torch.diff(measure_numbers,dim=1).shape)
    len_note = torch.LongTensor([measure_numbers.shape[1] * len(measure_numbers)]).to(measure_numbers.device)
  len_note[len_note==1] = measure_numbers.shape[1]

  return len_note
