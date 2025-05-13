from torch import nn

class PositionwiseFeedForwardLayer(nn.Module):
    # hidden_dim : embedding dimension of a word
    # pf_dim : inner embedding dimension in feed forward layer
    # dropout_ratio
    def __init__(self, hidden_dim, pf_dim, dropout_ratio,output_dim=None):
        super().__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        if output_dim != None:
            self.fc_2 = nn.Linear(pf_dim, output_dim)
        else:
            self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x):
        # x : [batch_size, seq_len, hidden_dim]
        # → x : [batch_size, seq_len, pf_dim]
        x = self.dropout(self.gelu(self.fc_1(x)))
        # → x : [batch_size, seq_len, hidden_dim]
        x = self.fc_2(x)
        
        return x