import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers

    """
    def __init__(self, n_items, M2, hidden_size, embedding_dim, batch_size, n_layers = 1):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.M2 = M2
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size+50  , bias=False)
        
        #self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        
        #len(seq):19
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))  # 19 x 512 x 50
        embs = pack_padded_sequence(embs, lengths)
        
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)
        
        # fetch the last hidden state of last timestamp
        ht = hidden[-1] # 19 x 512 x 200
        
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht # 512 x 200
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t) # 512 x 200

        
        # trans_vector 생성
        seq2= torch.transpose(seq, 1, 0)
        if len( (seq2[0]==0).nonzero() )==0:
            if str(seq2[0][-1].item()) in self.M2.columns:
                trans_vec = (torch.FloatTensor(self.M2[str(seq2[0][-1].item())].values)).view(1,self.n_items)
            else:
                trans_vec=(torch.FloatTensor([0]*self.n_items)).view(1,self.n_items)
        else:
            index= (seq2[0]==0).nonzero()[0].item() -1
            if str(seq2[0][index].item()) in self.M2.columns:
                trans_vec = (torch.FloatTensor(self.M2[str(seq2[0][index].item())].values)).view(1,self.n_items)
            else:
                trans_vec=(torch.FloatTensor([0]*self.n_items)).view(1,self.n_items)
                    
        for i in seq2[1:]:
            if len( (i==0).nonzero() )==0:
                if str(i[-1].item()) in self.M2.columns:
                    trans_vec = torch.cat( (trans_vec, ((torch.FloatTensor(self.M2[str(i[-1].item())].values)).view(1,self.n_items)) ) , 0)
                else:
                    trans_vec= torch.cat( ( trans_vec, ((torch.FloatTensor([0]*self.n_items)).view(1,self.n_items)) ) , 0)
            
            else:
                index= (i==0).nonzero()[0].item() -1
                
                if str(i[index].item()) in self.M2.columns:
                    trans_vec = torch.cat( (trans_vec, ((torch.FloatTensor(self.M2[str(i[index].item())].values)).view(1,self.n_items)) ) , 0)
                else:
                    trans_vec= torch.cat( ( trans_vec, ((torch.FloatTensor([0]*self.n_items)).view(1,self.n_items)) ) , 0)
        
        trans_vec = trans_vec.to(self.device) # 512 x n_items
        
        item_embs = self.emb(torch.arange(self.n_items).to(self.device)) # n_items x 50
        
        
        trans_emb = torch.matmul(trans_vec, item_embs) # 512 x 50
        
        final_representation = torch.cat( (c_t, trans_emb) , 1) # 512 x 250
        scores = torch.matmul(final_representation, self.b(item_embs).permute(1, 0))  # 512 x 250 , 250 x n_items = 512 x n_items
        
        
        
        #scores1 = torch.matmul(c_t, self.b(item_embs).permute(1, 0)) #512 x 200 , 200 x n_items = 512 x n_items
        #scores2 = torch.matmul(trans_emb, item_embs.permute(1, 0)) # 512 x 50 , 50 x n_items = 512 x n_items
        # scores = self.sf(scores)
        #scores= scores1+scores2
        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        
