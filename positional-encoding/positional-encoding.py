import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len,d_model))

    # Making columns : these will be the pos value in the formula since they contain the token index
    columns = np.arange(seq_len).reshape((seq_len,1))
    # Making rows : these will be the i value in the formula since they contain the dim index
    # We're adding 1+d/2 which will will always truncate for odd dim
    # For even dim, we will need that extra dimension of sin
    rows = np.arange(int(d_model/2)+1).reshape((1,(1+int(d_model/2))))

    # Updating odd dims
    pe[:,1::2] = np.cos(columns/base**(2*rows/d_model))[:,:-1]

    # Updating even dims
    if d_model%2==0: # Truncate if d_model is even
        pe[:,0::2] = np.sin(columns/base**(2*rows/d_model))[:,:-1]
    else: # Keep the extra dim if d_model is odd
        pe[:,0::2] = np.sin(columns/base**(2*rows/d_model))

    return pe