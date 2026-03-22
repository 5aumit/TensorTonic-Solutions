import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Define max length
    if max_len is None:
        max_len = max([len(seq) for seq in seqs])

    for i,seq in enumerate(seqs):
        #If equal -> skip
        if len(seq)==max_len:
            continue
        #If more -> truncate
        elif len(seq)>max_len:
            seqs[i] = seqs[i][:max_len]
        #If less -> pad
        elif len(seq)<max_len:
            seqs[i] = seqs[i]+[pad_value]*(max_len - len(seq))

    return seqs