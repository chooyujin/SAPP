from torch import nn, einsum
import torch.nn.functional as F
import torch

class MultiHeadAdjAttentionLayer(nn.Module):
    # hidden_dim : embedding dimension of word
    # n_head : number of heads
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        assert(hidden_dim % n_heads == 0), "hidden_dim size needs to be div by n_heads"
        
        self.hidden_dim = hidden_dim 
        self.n_heads = n_heads
        self.head_dim = hidden_dim // self.n_heads
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim) 
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.str_fc_k = nn.Linear(hidden_dim,128)
        self.str_fc_v = nn.Linear(hidden_dim,128)
        
        self.fc_o = nn.Linear(hidden_dim+128, hidden_dim)
        
        self.null_k = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.null_v = nn.Parameter(torch.randn(n_heads, self.head_dim))
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.str_scale =torch.sqrt(torch.FloatTensor([1])).to(device)

    def forward(self, query, key, value,x_rsa, mask=None,rsa_mask=None):
        
        batch_size = query.shape[0]
        
        # query : [batch_size, query_len, hidden_dim] → Q : [batch_size, query_len, hidden_dim]
        Q = self.fc_q(query)
        # key : [batch_size, key_len, hidden_dim] → K : [batch_size, key_len, hidden_dim]
        K = self.fc_k(key)
        # value : [batch_size, value_len, hidden_dim] → V : [batch_size, value_len, hidden_dim]
        V = self.fc_v(value)
        
        str_K = self.str_fc_k(key)
        str_V = self.str_fc_k(value)
        
        rsa_Q = x_rsa.reshape(batch_size,-1,1, 128).permute(0,2,1,3)
        
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        str_K = str_K.reshape(batch_size,-1,1,128).permute(0,2,1,3)
        str_V = str_V.reshape(batch_size,-1,1,128).permute(0,2,1,3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) 
        rsa_energy = torch.matmul(rsa_Q,str_K.permute(0,1,3,2))
        if mask != None:
            energy = energy.masked_fill(mask==0, -1e10)
            rsa_energy = rsa_energy.masked_fill(mask==0,-1e10)
            
        if rsa_mask != None:
            rsa_energy = rsa_energy.masked_fill(rsa_mask==0,-1e10)
            
        # calculate attention score : probability about each data(ex.word)
        # We get attention score based on key_len.
        # attention : [batch_size, n_heads, query_len, key_len]
        energy = energy / self.scale
        rsa_energy = rsa_energy / self.str_scale
        
        attention = torch.softmax(energy, dim=-1)
        rsa_attention = torch.softmax(rsa_energy,dim=-1)
        
        # attention : [batch_size, n_heads, query_len, key_len]
        # V : [batch_size, n_heads, value_len, head_dim]
        # x : [batch_size, n_heads, query_len, head_dim] (∵ key_len = value_len in this self-attention)
        x = torch.matmul(self.dropout(attention), V)
        rsa_x = torch.matmul(self.dropout(rsa_attention), str_V)
        
        # x : [batch_size, n_heads, query_len, head_dim]
        # → x : [batch_size, query_len, n_heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size,-1,self.hidden_dim)
        rsa_x = rsa_x.permute(0,2,1,3).contiguous().reshape(batch_size,-1,128)
        
        #x = torch.concat((x,rsa_x,plddt_x,disorder_x),dim=-1)
        x = torch.concat((x,rsa_x),dim=-1)
        # x : [batch_size, query_len, hidden_dim]
        
        # x : [batch_size, query_len, hidden_dim]
        x = self.fc_o(x)
        
        
        return x, [attention,rsa_attention]