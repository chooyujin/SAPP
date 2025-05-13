import torch.nn as nn
from .Attention import MultiHeadAdjAttentionLayer
from .FeedForward import PositionwiseFeedForwardLayer
from .Embedding import PositionalEmbedding, TokenEmbedding, BERTEmbedding
import torch 

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden_dim, attn_heads, feed_forward_hidden, dropout,device, attn_dropout=0.2):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.self_attention_layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.adj_attention = MultiHeadAdjAttentionLayer(hidden_dim, attn_heads, attn_dropout,device)
        self.adj_attention_layer_norm = nn.LayerNorm(hidden_dim)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hidden_dim, feed_forward_hidden, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x,x_rsa,mask,rsa_mask):

        _x,attns= self.adj_attention(x,x,x,x_rsa, mask=mask,rsa_mask=rsa_mask)
        
        x = self.adj_attention_layer_norm(x + self.dropout(_x))
        
        _x = self.positionwise_feedforward(x)
        
        x = self.feedforward_layer_norm(x + self.dropout(_x))
    
        return x,attns

class SAPP_Model(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, vocab_size,window,hidden,feed_forward_dim, n_layers, attn_heads,device, dropout=0.2):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.device = device
        self.window_size = window

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.RSA_linear = nn.Linear(21,128)
        
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, feed_forward_dim, dropout,device) for _ in range(n_layers)])
        
        self.linear_input = self.window_size*2+1
        self.output_linear = nn.Linear(self.linear_input*256,1)
        
    def forward(self,x,x_RSA,mask=None,rsa_mask=None):
        
        if mask != None:
            mask = mask.unsqueeze(1)
        if rsa_mask != None:
            rsa_mask = rsa_mask.unsqueeze(1)
            
        x = self.embedding(x)
        x_RSA = self.RSA_linear(x_RSA)
        layer1_attns = ''
        layer2_attns = ''
        # running over multiple transformer blocks
        for i, transformer in enumerate(self.transformer_blocks):
            x,attns = transformer.forward(x,x_RSA, mask,rsa_mask)
            if i == 0:
                layer1_attns = attns
            else:
                layer2_attns = attns
            
        
        x = x.reshape(-1,self.linear_input*256)
        
        output = self.output_linear(x)  

        return torch.sigmoid(output),[layer1_attns,layer2_attns]