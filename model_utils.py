import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



def find_boundaries(diff_boundary, measure_numbers, i):
  """ # diff_boundary [565, 2], measure_numbers [32, 192], i (int)
  Finds the boundaries for a specific batch index where a new measure starts.

  Args:
      diff_boundary (torch.Tensor): A tensor of shape `[total_note_tokens, 2]` where each element contains 
          the batch index and the last note's offset of a measure.
      measure_numbers (torch.Tensor): A zero-padded tensor of shape `[batch size, sequence length]` containing measure numbers.
      i (int): The batch index for which to find boundaries.

  Returns:
      out (List[int]): A list [start positions of a new measure in a batch],
          containing lists of positions indicating the start of a new measure for a given batch.
  """ # out : 0, 새로운 마디가 시작되는 인덱스들, 마지막 마디 인덱스
  out = [0] + (diff_boundary[diff_boundary[:,0]==i][:,1]+1 ).tolist() + [torch.max(torch.nonzero(measure_numbers[i])).item()+1]
  if out[1] == 0: # if the first boundary occurs in 0, it will be duplicated
    out.pop(0)
  return out


def find_boundaries_batch(measure_numbers):
  """ # measure_numbers : [32, 192]
  Identifies all measure boundaries for each batch.

  Args:
      measure_numbers (torch.Tensor): A zero-padded tensor of shape `[N, T]`, where `N` is the batch size 
          and `T` is the sequence length. Each element indicates the measure number.
          
  Returns:
      (List[List[int]]): A list contains `N`(batch size) lists [start positions of a new measure in each batch],
          where each inner list contains the start positions of new measure for each batch.
  """
  diff_boundary = torch.nonzero(measure_numbers[:,1:] - measure_numbers[:,:-1]).cpu() # *10) diff_boundary :[565,2] -> 각 element : [몇번째 배치, 해당 배치 내 새로운 마디가 시작되는 인덱스]
  return [find_boundaries(diff_boundary, measure_numbers, i) for i in range(len(measure_numbers))]


def get_softmax_by_boundary(similarity, boundaries, fn=torch.softmax):
  ''' # similarity[batch_idx] : [192, 8], boundaries[batch_idx] : list(리스트 길이는 매 인덱스마다 다름)
  similarity = similarity of a single sequence of data (T x C)
  boundaries = list of a boundary index (T)
  '''
  return  [fn(similarity[boundaries[i-1]:boundaries[i],: ], dim=0)  \
              for i in range(1, len(boundaries))
                if boundaries[i-1] < boundaries[i] # sometimes, boundaries can start like [0, 0, ...]
          ]


def make_higher_node(note_hidden, attention_weights, measure_numbers):
    """ # note_hidden [32, 192, 512], attention_weights : ContextAttention, measure_numbers [32, 192]
    Prepares inputs for the measure-level GRU by aggregating note-level hidden states using `context attention`.

    This function processes note-level hidden vectors using attention weights and measure boundary information
    to generate aggregated representations for each measure.
    These representations serve as inputs to the measure-level GRU.
    
    Process:
        1. note_hidden -> `ContextAttention.get_attention` -> similarity
        2. measure_numbers -> `find_boundaries_batch` -> boundaries
        3. similarity, boundaries -> `get_softmax_by_boundary` -> softmax_similarity
        4. softmax_similarity, note_hidden -> (return) higher_nodes
    
    Args:
        note_hidden (torch.Tensor): 
            Note-level hidden states of shape `[batch_size, sequence_length, hidden_size]`
        attention_weights (ContextAttention): 
            The attention mechanism used to compute similarity scores and weight the note-level hidden states.
        measure_numbers (torch.Tensor): 
            Tensor indicating measure boundaries for each batch, with shape `[batch_size, sequence_length]`.

    Returns:
        higher_nodes (torch.Tensor): 
            Measure-level input representations of shape `[batch_size, num_measures, hidden_size]`
    """
    similarity = attention_weights.get_attention(note_hidden) # *7) similarity : [32, 192, 8]
    boundaries = find_boundaries_batch(measure_numbers) # *9) len(boundaries) : 32 / 각 배치마다 마디 수는 다르므로, boundaries 내 리스트들도 길이가 제각각
    softmax_similarity = torch.nn.utils.rnn.pad_sequence(
      [torch.cat(get_softmax_by_boundary(similarity[batch_idx], boundaries[batch_idx])) # *11)
        for batch_idx in range(len(note_hidden))], batch_first=True) # [32, 192, 8]
    
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.stack(note_hidden.split(split_size=attention_weights.head_size, dim=2), dim=2) # [32, 192, 8, 64]
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1]) # [32, 192, 8, 64] * [32, 192, 8, 64] = [32, 192, 8, 64]
        weighted_x = weighted_x.view(x_split.shape[0], x_split.shape[1], note_hidden.shape[-1]) # [32, 192, 8, 64] -> [32, 192, 512] : 8*64 -> 512
        higher_nodes = torch.nn.utils.rnn.pad_sequence([
          torch.cat([torch.sum(weighted_x[i:i+1,boundaries[i][j-1]:boundaries[i][j],: ], dim=1) for j in range(1, len(boundaries[i]))], dim=0) \
          for i in range(len(note_hidden))], batch_first=True)
    else:
        weighted_sum = softmax_similarity * note_hidden
        higher_nodes = torch.cat([torch.sum(weighted_sum[:,boundaries[i-1]:boundaries[i],:], dim=1) 
                                for i in range(1, len(boundaries))]).unsqueeze(0)
    return higher_nodes


def span_measure_to_note_num(measure_hidden, measure_number):
  """
  Broadcasts measure-level output to note-level, assigning each note the hidden output of the corresponding previous measure.

  This function transforms measure-level hidden states into note-level representations,
  so that each note or token is associated with the measure-level output of the preceding measure.

  Args:
      measure_hidden (torch.Tensor): Tensor of shape `[N, T_measure, C]`
      measure_number (torch.Tensor): Tensor of shape `[N, T_note]`

  Returns:
      torch.Tensor: A tensor of shape `[N, T_note, C]` containing note-level representations derived from measure-level outputs.
  """
  zero_shifted_measure_number = measure_number - measure_number[:,0:1]
  len_note = cal_length_from_padded_measure_numbers(measure_number)

  batch_indices = torch.cat([torch.ones(length)*i for i, length in enumerate(len_note)]).long()
  note_indices = torch.cat([torch.arange(length) for length in len_note])
  measure_indices = torch.cat([zero_shifted_measure_number[i,:length] for i, length in enumerate(len_note)]).long()

  measure_indices = measure_indices - 1 # note has to get only previous measure info

  span_mat = torch.zeros(measure_number.shape[0], measure_number.shape[1], measure_hidden.shape[1]).to(measure_hidden.device)
  span_mat[batch_indices, note_indices, measure_indices] = 1 # assigning a value of 1 to valid indice
  span_mat[:, :, -1] = 0 # last measure is not used
  spanned_measure = torch.bmm(span_mat, measure_hidden) # spanning measures to notes
  return spanned_measure


def cal_length_from_padded_measure_numbers(measure_numbers):
  """
  Calculates the length of note sequences from zero-padded measure numbers.

  Args:
      measure_numbers (torch.Tensor): Tensor of shape `[batch size, sequence length]`, zero-padded measure number.

  Returns:
      len_note (torch.Tensor): Tensor of shape `[batch size]` containing the effective lengths of the note sequences in the batch.
  """
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
