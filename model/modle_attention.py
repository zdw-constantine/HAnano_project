from torch import nn
import torch
import torch.nn.functional as F

class Trend_aware_attention(nn.Module):
    '''
    Trend_aware_attention  mechanism
    X:      [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, kernel_size):
        super(Trend_aware_attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D,D)
        self.FC = nn.Linear(D,D)
        self.kernel_size = kernel_size
        self.padding = self.kernel_size-1
        self.cnn_q = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)
    def forward(self, X):
        
        batch_size = X.shape[0]

        X_ = X.permute(0, 3, 2, 1) 

        # key: (B,T,N,D)  value: (B,T,N,D)
        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1) 
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1) 
        value = self.FC_v(X) 

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0) 
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0) 

        query = query.permute(0, 2, 1, 3)  # query: (B*k,T,N,d) --> (B*k,N,T,d)
        key = key.permute(0, 2, 3, 1)      # key: (B*k,T,N,d) --> (B*k,N,d,T)
        value = value.permute(0, 2, 1, 3)  # key: (B*k,T,N,d) --> (B*k,N,T,d)

        attention = (query @ key) * (self.d ** -0.5) #  (B*k,N,T,d) @ (B*k,N,d,T) = (B*k,N,T,T)
        attention = F.softmax(attention, dim=-1) 

        X = (attention @ value) # (B*k,N,T,T) @ (B*k,N,T,d) = (B*k,N,T,d)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1) # (B*k,N,T,d)-->(B,N,T,d*k)==(B,N,T,D)
        X = self.FC(X) # 
        return X.permute(0, 2, 1, 3) # (B,N,T,D)-->(B,T,N,D)

class gatedFusion(nn.Module):

    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        x11 = self.fc1(x1)
        x22 = self.fc2(x2)
        
        z = torch.sigmoid(x11+x22)
        
        out = z*x1 + (1-z)*x2
        return out


class AttentionLayer(nn.Module):
    def __init__(self, K, d, kernel_size):
        super(AttentionLayer, self).__init__()
        self.Model=Trend_aware_attention(K=K, d=d, kernel_size=kernel_size)
        self.gate=gatedFusion(dim=324)

    def forward(self, X):
        x=X
        #X-(B-16,C/D-256,T-400)
        X = X.unsqueeze(2)
        #X-(B-16,C/D-256,n-1,T-400)
        X = X.permute(0,3,2,1)
        #X-(B-16,t-400,n-1,c-64)
        #print(X.shape)
        out = self.Model(X)
        #out-(B-16,t-400,n-1,c-64)
        out = out.squeeze(2)
        #print(out.shape)
        #print(x.shape)
        x=x.permute(0,2,1)
        out = self.gate(out,x)
        out = out.permute(0,2,1)
        #OUT-(B-16,C/D-512,T-400)
        #out-(t-400,B-16,C-512)
        #print(out.shape)
        return out
