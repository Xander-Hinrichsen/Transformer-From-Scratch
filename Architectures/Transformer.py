# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.334795Z","iopub.execute_input":"2023-02-24T04:47:32.335282Z","iopub.status.idle":"2023-02-24T04:47:32.341575Z","shell.execute_reply.started":"2023-02-24T04:47:32.335244Z","shell.execute_reply":"2023-02-24T04:47:32.340289Z"}}
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.343414Z","iopub.execute_input":"2023-02-24T04:47:32.343840Z","iopub.status.idle":"2023-02-24T04:47:32.355722Z","shell.execute_reply.started":"2023-02-24T04:47:32.343805Z","shell.execute_reply":"2023-02-24T04:47:32.354398Z"}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.358160Z","iopub.execute_input":"2023-02-24T04:47:32.358858Z","iopub.status.idle":"2023-02-24T04:47:32.369704Z","shell.execute_reply.started":"2023-02-24T04:47:32.358819Z","shell.execute_reply":"2023-02-24T04:47:32.368659Z"}}
#have to create our own cusomtom Softmax because of issues of division by zero when softmaxing
#on the rows that are only padding (only -inf)
def customSoftmax3dfunc(x):
    ##get each row max and subtract row by it, except if max is inf
    maxes, idx = torch.max(x, dim=2, keepdim=True)
    maxes[maxes==-torch.inf] = 0
    x -= maxes
    xexp = torch.exp(x)
    return xexp / (torch.sum(xexp, dim=2, keepdim=True) + 1e-10)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.371379Z","iopub.execute_input":"2023-02-24T04:47:32.371934Z","iopub.status.idle":"2023-02-24T04:47:32.383236Z","shell.execute_reply.started":"2023-02-24T04:47:32.371870Z","shell.execute_reply":"2023-02-24T04:47:32.382327Z"}}
class customSoftmax3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = customSoftmax3dfunc
    def forward(self, xb):
        return self.softmax(xb)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.386188Z","iopub.execute_input":"2023-02-24T04:47:32.387423Z","iopub.status.idle":"2023-02-24T04:47:32.399006Z","shell.execute_reply.started":"2023-02-24T04:47:32.387353Z","shell.execute_reply":"2023-02-24T04:47:32.397744Z"}}
def positional_encoding(seq_len, dmodel, device=device):
    #shape of output
    PE = torch.ones(seq_len, dmodel, dtype=torch.float32).to(device)
    #positional encoding
    pos = torch.arange(seq_len).view(-1,1).to(device)
    PE *= pos

    #10000^2i/dmodel
    poly = torch.ones(1,dmodel).to(device)
    poly[:, 2*torch.arange((dmodel+1)//2)] = (2*torch.arange((dmodel+1)//2, dtype=torch.float32) / dmodel).to(device)
    poly[:, 2*torch.arange(dmodel//2) + 1] = (2*torch.arange((dmodel//2), dtype=torch.float32) / dmodel).to(device)
    poly = 10000**poly
    PE /= poly.view(1,-1)

    ##sin even colomns, cos odd columns
    PE[:, 2*torch.arange((dmodel + 1)//2)] = torch.sin(PE[:, 2*torch.arange((dmodel+1)//2)])
    PE[:, 2*torch.arange((dmodel)//2) + 1] = torch.cos(PE[:, 2*torch.arange((dmodel)//2) + 1])
    return PE
#positional_encoding(4, 512)[:4,:4]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.400383Z","iopub.execute_input":"2023-02-24T04:47:32.400943Z","iopub.status.idle":"2023-02-24T04:47:32.411574Z","shell.execute_reply.started":"2023-02-24T04:47:32.400909Z","shell.execute_reply":"2023-02-24T04:47:32.410559Z"}}
def add_pe(xb, device=device):
    return xb + positional_encoding(xb.shape[1], xb.shape[2], device=device)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.412994Z","iopub.execute_input":"2023-02-24T04:47:32.413319Z","iopub.status.idle":"2023-02-24T04:47:32.425197Z","shell.execute_reply.started":"2023-02-24T04:47:32.413291Z","shell.execute_reply":"2023-02-24T04:47:32.424188Z"}}
def add_padded_pe(xb, pad_idxs, device=device):
    PE = positional_encoding(xb.shape[1], xb.shape[2], device=device)
    xb_pe = xb+PE
    mask = torch.arange(xb.shape[1], dtype=torch.float32).to(device) < pad_idxs.reshape(-1, 1, 1)
    return xb_pe * mask.reshape(xb.shape[0],-1,1).to(device)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.426943Z","iopub.execute_input":"2023-02-24T04:47:32.427629Z","iopub.status.idle":"2023-02-24T04:47:32.442504Z","shell.execute_reply.started":"2023-02-24T04:47:32.427583Z","shell.execute_reply":"2023-02-24T04:47:32.441609Z"}}
#test = torch.zeros(2,4,4)
#pad_idxs = torch.LongTensor([2,3])
#add_padded_pe(test, pad_idxs)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.443933Z","iopub.execute_input":"2023-02-24T04:47:32.444469Z","iopub.status.idle":"2023-02-24T04:47:32.458178Z","shell.execute_reply.started":"2023-02-24T04:47:32.444436Z","shell.execute_reply":"2023-02-24T04:47:32.456828Z"}}
##if making mixed mask - it should be in order of q, k
def make_pad_masks(pad_idxs, max_seq_len, pad_idxs2=None, max_seq_len2=None, device=device):
    if pad_idxs2 == None and max_seq_len2 == None: 
        pad_masks = torch.ones(pad_idxs.shape[0], max_seq_len, max_seq_len).to(device)
        mask = ((torch.arange(pad_masks.shape[1], dtype=torch.float32).to(device) >= pad_idxs.reshape(-1, 1, 1)))
        # apply the mask to the array
        mask = mask.to(torch.float32).to(device)
        mask[mask == 1] = -torch.inf
        return (pad_masks * (mask.reshape(pad_masks.shape[0],-1,1) + mask.reshape(pad_masks.shape[0],1,-1)))
    elif pad_idxs2 != None and max_seq_len2 != None:
        pad_masks = torch.ones(pad_idxs.shape[0], max_seq_len, max_seq_len2).to(device)
        q_mask = torch.arange(pad_masks.shape[1], dtype=torch.float32).to(device) >= pad_idxs.reshape(-1,1,1)
        k_mask = torch.arange(pad_masks.shape[2], dtype=torch.float32).to(device) >= pad_idxs2.reshape(-1,1,1)
        q_mask = q_mask.to(torch.float32).to(device)
        k_mask = k_mask.to(torch.float32).to(device)
        q_mask[q_mask==1] = -torch.inf
        k_mask[k_mask==1] = -torch.inf
        return (pad_masks * (q_mask.reshape(pad_masks.shape[0],-1,1) + k_mask.reshape(pad_masks.shape[0],1,-1)))
    else:
        print('error, incorrect use of make_pad_masks args')
#make_pad_masks(torch.LongTensor([3,5]), 7, torch.LongTensor([4,6]),9)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.461533Z","iopub.execute_input":"2023-02-24T04:47:32.462154Z","iopub.status.idle":"2023-02-24T04:47:32.470421Z","shell.execute_reply.started":"2023-02-24T04:47:32.462118Z","shell.execute_reply":"2023-02-24T04:47:32.469238Z"}}
def make_result_pad_masks(pad_idxs, seq_size,vocab_size, device=device):
    pad_masks = torch.ones(pad_idxs.shape[0], seq_size, vocab_size).to(device)
    mask = ((torch.arange(pad_masks.shape[1], dtype=torch.float32).to(device) >= pad_idxs.reshape(-1, 1, 1)))
    mask = mask.to(torch.float32).to(device)
    mask[mask == 1] = -torch.inf
    result = (pad_masks * (mask.reshape(pad_masks.shape[0],-1,1) ))
    bad_indices = result[:,:,0] != -torch.inf 
    result[:, :, 0] = 0
    result[:,:,0][bad_indices] = -torch.inf
    
    return result
#asdf = make_result_pad_masks(torch.LongTensor([3,1]), 5,7)
#asdf

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.471655Z","iopub.execute_input":"2023-02-24T04:47:32.472245Z","iopub.status.idle":"2023-02-24T04:47:32.486401Z","shell.execute_reply.started":"2023-02-24T04:47:32.472211Z","shell.execute_reply":"2023-02-24T04:47:32.485279Z"}}
#nn.Softmax(dim=2)(asdf)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.488223Z","iopub.execute_input":"2023-02-24T04:47:32.489206Z","iopub.status.idle":"2023-02-24T04:47:32.498933Z","shell.execute_reply.started":"2023-02-24T04:47:32.489156Z","shell.execute_reply":"2023-02-24T04:47:32.497814Z"}}
#customSoftmax3d()(asdf)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.500695Z","iopub.execute_input":"2023-02-24T04:47:32.501324Z","iopub.status.idle":"2023-02-24T04:47:32.515102Z","shell.execute_reply.started":"2023-02-24T04:47:32.501289Z","shell.execute_reply":"2023-02-24T04:47:32.513933Z"}}
class AttentionHead(nn.Module):
    def __init__(self, dk=64, dv=64, dmodel=512, masked=False, device=device):
        super().__init__()
        ##to project the inputs into q,k,v
        self.query_projector = nn.Linear(dmodel, dk)
        self.key_projector = nn.Linear(dmodel, dk)
        self.value_projector = nn.Linear(dmodel, dv)
        self.device = device
        ##for the scaling
        self.scalar = (torch.tensor([1]) / torch.sqrt(torch.tensor(dk))).to(device)
        
        ##normalization
        self.softmax = customSoftmax3d()
        
        ##only make upper right triangle -inf if masked attention
        self.masked = masked
    def forward(self, q, k, v, pad_masks):
        ##project q,k,v into dmodel/num_heads # of dimensions
        q = self.query_projector(q)
        k = self.key_projector(k)
        v = self.value_projector(v)
        
        ##matmul q and k.t()
        qkT = torch.bmm(q, torch.transpose(k,1,2))

        ##scale the output by dividing by sqrt(dmodel)
        qkT = self.scalar * qkT
        ##now we need to handle the masking if this is a masked head
        mask = torch.zeros((qkT.shape[1],qkT.shape[2])).to(self.device)
        if self.masked:
            indices = torch.triu_indices(qkT.shape[1],qkT.shape[2], offset=1).to(self.device)
            mask[indices[0], indices[1]] = -torch.inf
            #print(mask)
        ##now we can softmax
        attention_filter = self.softmax((qkT + pad_masks) + mask)
        #never gonna use the mask again
        del(mask)
        
        ##return attentionfilter @ values - these are the new, mutated words
        ##the resultant matrix should be of shape #words x dv
        return (torch.bmm(attention_filter, v))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.517063Z","iopub.execute_input":"2023-02-24T04:47:32.517988Z","iopub.status.idle":"2023-02-24T04:47:32.535637Z","shell.execute_reply.started":"2023-02-24T04:47:32.517936Z","shell.execute_reply":"2023-02-24T04:47:32.534664Z"}}
q = torch.ones(1,4,512)
k = torch.ones(1,4,512)
qkT = torch.bmm(q, torch.transpose(k,1,2))
mask = torch.zeros((qkT.shape[1],qkT.shape[2]))
indices = torch.triu_indices(qkT.shape[1],qkT.shape[2], offset=1)
mask[indices[0], indices[1]] = -torch.inf
qkT

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.537086Z","iopub.execute_input":"2023-02-24T04:47:32.538237Z","iopub.status.idle":"2023-02-24T04:47:32.545573Z","shell.execute_reply.started":"2023-02-24T04:47:32.538190Z","shell.execute_reply":"2023-02-24T04:47:32.544398Z"}}
mask

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.547135Z","iopub.execute_input":"2023-02-24T04:47:32.547503Z","iopub.status.idle":"2023-02-24T04:47:32.558965Z","shell.execute_reply.started":"2023-02-24T04:47:32.547471Z","shell.execute_reply":"2023-02-24T04:47:32.557741Z"}}
qkT + mask

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.560474Z","iopub.execute_input":"2023-02-24T04:47:32.560809Z","iopub.status.idle":"2023-02-24T04:47:32.573810Z","shell.execute_reply.started":"2023-02-24T04:47:32.560780Z","shell.execute_reply":"2023-02-24T04:47:32.572760Z"}}
qkT = torch.ones(1,5,7)
mask = torch.zeros((qkT.shape[1],qkT.shape[2]))
indices = torch.triu_indices(qkT.shape[1],qkT.shape[1], offset=1)
mask[indices[0], indices[1]] = -torch.inf
qkT

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.575514Z","iopub.execute_input":"2023-02-24T04:47:32.576342Z","iopub.status.idle":"2023-02-24T04:47:32.582719Z","shell.execute_reply.started":"2023-02-24T04:47:32.576292Z","shell.execute_reply":"2023-02-24T04:47:32.581769Z"}}
#mask

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.588394Z","iopub.execute_input":"2023-02-24T04:47:32.589419Z","iopub.status.idle":"2023-02-24T04:47:32.601791Z","shell.execute_reply.started":"2023-02-24T04:47:32.589370Z","shell.execute_reply":"2023-02-24T04:47:32.600459Z"}}
#need to implement parallel gpu
class MultiAttentionHead(nn.Module):
    def __init__(self, dk=64, dv=64, dmodel=512, num_heads=8, masked=False, device=device):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.num_heads = num_heads
        
        ##list of attentionheads
        self.AttentionHeads = nn.ModuleList()
        for i in range(self.num_heads):
            self.AttentionHeads.append(AttentionHead(dk=dk, dv=dv, dmodel=dmodel, masked=masked, device=device))
        
        ##linearl
        self.linear = nn.Linear(num_heads * dv, dmodel)
        
    def forward(self, q, k, v, pad_masks):
        out = self.AttentionHeads[0](q,k,v, pad_masks)
        ##concat
        for i in range(1, self.num_heads):
            ##concat along the last column - to restore original input shape
            out = torch.cat((out, self.AttentionHeads[i](q,k,v,pad_masks)), dim=2)
        
        #linear
        out = self.linear(out)
        return out

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.603626Z","iopub.execute_input":"2023-02-24T04:47:32.604397Z","iopub.status.idle":"2023-02-24T04:47:32.614111Z","shell.execute_reply.started":"2023-02-24T04:47:32.604350Z","shell.execute_reply":"2023-02-24T04:47:32.612838Z"}}
class FeedForward(nn.Module):
    def __init__(self, hidden_size=2048, dmodel=512):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dmodel, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dmodel))
    def forward(self, xb):
        return self.ff(xb)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.615543Z","iopub.execute_input":"2023-02-24T04:47:32.615868Z","iopub.status.idle":"2023-02-24T04:47:32.628396Z","shell.execute_reply.started":"2023-02-24T04:47:32.615839Z","shell.execute_reply":"2023-02-24T04:47:32.627236Z"}}
class TransformerBlock(nn.Module):
    def __init__(self, dk=64, dv=64, dmodel=512, num_heads=8, ff_hidden=2048, masked=False, device=device):
        super().__init__()
        self.MultiHeadAttention = MultiAttentionHead(dk=dk, dv=dv, dmodel=dmodel, num_heads=num_heads, masked=masked, device=device)
        self.layer_norm1 = nn.LayerNorm(dmodel)
        self.ff = FeedForward(hidden_size=ff_hidden, dmodel=dmodel)
        self.layer_norm2 = nn.LayerNorm(dmodel)
    def forward(self, q, k, v, pad_masks):
        ##defaulting to adding v as residual everytime, becuase this is the case
        attended_to = self.MultiHeadAttention(q,k,v,pad_masks)
        normed1 = self.layer_norm1(attended_to + q)
        fed_forward = self.ff(normed1)
        normed2 = self.layer_norm2(fed_forward + normed1)
        return normed2

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.629834Z","iopub.execute_input":"2023-02-24T04:47:32.630247Z","iopub.status.idle":"2023-02-24T04:47:32.646356Z","shell.execute_reply.started":"2023-02-24T04:47:32.630215Z","shell.execute_reply":"2023-02-24T04:47:32.644776Z"}}
class Encoder(nn.Module):
    def __init__(self, dk=64, dv=64, dmodel=512, num_heads=8, ff_hidden=2048, num_blocks=6,device=device):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.num_blocks = num_blocks
        for i in range(self.num_blocks):
            self.encoder.append(TransformerBlock(dk=dk, dv=dv, dmodel=dmodel, num_heads=num_heads, ff_hidden=ff_hidden,device=device))
    def forward(self, xb, pad_masks):
        for i in range(self.num_blocks):
            xb = self.encoder[i](xb,xb,xb, pad_masks)
        return xb

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.648013Z","iopub.execute_input":"2023-02-24T04:47:32.648376Z","iopub.status.idle":"2023-02-24T04:47:32.660117Z","shell.execute_reply.started":"2023-02-24T04:47:32.648345Z","shell.execute_reply":"2023-02-24T04:47:32.658918Z"}}
class BiggerTransformerBlock(nn.Module):
    def __init__(self, dk=64, dv=64, dmodel=512, num_heads=8, ff_hidden=2048, device=device):
        super().__init__()
        self.maskedAttentionHead = MultiAttentionHead(dk=dk, dv=dv, dmodel=dmodel, num_heads=num_heads, masked=True, device=device)
        self.layer_norm = nn.LayerNorm(dmodel)
        self.regular_block = TransformerBlock(dk=dk, dv=dv, dmodel=dmodel, num_heads=num_heads, 
                                              ff_hidden=ff_hidden, masked=False, device=device)
    def forward(self,xb,pad_masks,encoder_output,mixed_pad_masks):
        out = self.maskedAttentionHead(xb,xb,xb,pad_masks)
        out = self.layer_norm(out + xb)
        ##q is the output of the masked attention, k and v are the encoder output
        out = self.regular_block(out,encoder_output,encoder_output,mixed_pad_masks)
        return out

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.662297Z","iopub.execute_input":"2023-02-24T04:47:32.663131Z","iopub.status.idle":"2023-02-24T04:47:32.675667Z","shell.execute_reply.started":"2023-02-24T04:47:32.663080Z","shell.execute_reply":"2023-02-24T04:47:32.674714Z"}}
class Decoder(nn.Module):
    def __init__(self, dk=64, dv=64, dmodel=512, num_heads=8, ff_hidden=2048, num_blocks=6,device=device):
        super().__init__()
        self.decoder = nn.ModuleList()
        self.num_blocks = num_blocks
        for i in range(self.num_blocks):
            self.decoder.append(BiggerTransformerBlock(dk=dk,dv=dv,dmodel=dmodel,num_heads=num_heads,
                                                       ff_hidden=ff_hidden,device=device))
    def forward(self,xb, pad_masks, encoder_output, mixed_pad_masks):
        for i in range(self.num_blocks):
            xb = self.decoder[i](xb,pad_masks,encoder_output, mixed_pad_masks)
        return xb

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.677566Z","iopub.execute_input":"2023-02-24T04:47:32.678341Z","iopub.status.idle":"2023-02-24T04:47:32.706514Z","shell.execute_reply.started":"2023-02-24T04:47:32.678294Z","shell.execute_reply":"2023-02-24T04:47:32.705505Z"}}
##this assumes that pad_idx, sos_token, and eos_token are already indices 0,1,2 respectively
class Transformer(nn.Module):
    def __init__(self, vocab, vocab_hashtable, padding_idx=0, dk=64, dv=64, dmodel=512, 
                 device=device, num_heads=8, ff_hidden=2048, num_blocks=6):
        super().__init__()
        ##assumtion is that vocab_size includes padding, sos, and eos
        self.vocab = vocab
        self.vocab_hashtable = vocab_hashtable
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.vocab_size = len(vocab)
        self.start_token_idx = 1
        self.end_token_idx = 2
        self.device = device
        
        self.embedding = nn.Embedding(self.vocab_size, dmodel, padding_idx=padding_idx)
        self.add_pe = add_pe
        self.add_padded_pe = add_padded_pe
        self.make_pad_masks = make_pad_masks
        self.make_result_pad_masks = make_result_pad_masks
        self.encoder = Encoder(dk=dk,dv=dv,dmodel=dmodel,num_heads=num_heads,
                               ff_hidden=ff_hidden,num_blocks=num_blocks,device=device)
        self.decoder = Decoder(dk=dk,dv=dv,dmodel=dmodel,num_heads=num_heads,
                               ff_hidden=ff_hidden,num_blocks=num_blocks,device=device)
        
        self.softmax = customSoftmax3d()
    
    ##assume that the yb already has the start token attached to the front
    ##output comes out as (batch,token, class_prob)
    ##for ce loss - need to permute to (batch, class_prob, token)
    def forward(self,xb, xb_pad_idxs, yb, yb_pad_idxs):
        ############################################################
        ##input to the encoder
        xb = self.embedding(xb)
        xb = self.add_padded_pe(xb, xb_pad_idxs, device=self.device)
        ###max sequence length is xb.shape[1] - number of rows is
        ###consistent across entire batch bc of padding
        xb_pad_masks = self.make_pad_masks(xb_pad_idxs, xb.shape[1], device=self.device)
        encoded = self.encoder(xb, xb_pad_masks)
        ###########################################################
        
        ###########################################################
        ##input to the decoder
        yb = self.embedding(yb)
        yb = self.add_padded_pe(yb, yb_pad_idxs, device=self.device)
        yb_pad_masks = self.make_pad_masks(yb_pad_idxs, yb.shape[1], device=self.device)
        mixed_pad_masks = self.make_pad_masks(yb_pad_idxs, yb.shape[1], xb_pad_idxs, xb.shape[1], device=self.device)
        decoded = self.decoder(yb, yb_pad_masks, encoded, mixed_pad_masks)
        ###########################################################
        #finally, we broadcast decoded @ embedding.weights.t()
        #projects into the dimension the same as the vocab size
        out = decoded @ self.embedding.weight.t()
        
        ##we don't need to softmax because ce does it for you
        #out = self.softmax(out)
        ##delete interim additives to model
        del(xb_pad_masks); del(yb_pad_masks); del(mixed_pad_masks)
        result_pad_masks = self.make_result_pad_masks(yb_pad_idxs, out.shape[1], out.shape[2], device=self.device)
        out += result_pad_masks
        del(result_pad_masks)
        return out
        
    ##this is its own beast 
    def make_inference(self, sequence, max_text_len):
        with torch.no_grad():
            xb = self.tokenize_xb(sequence)
            #print('tokenized', xb)
            xb = self.embedding(xb)
            xb = self.add_pe(xb, device=self.device)
            #make false pad masks that are out of range - so all 0's pad mask
            xb_pad_idxs = (torch.ones(1).long()*xb.shape[1]*10000).to(self.device)
            xb_pad_masks = self.make_pad_masks(xb_pad_idxs, xb.shape[1], device=self.device)
            encoded = self.encoder(xb, xb_pad_masks)
            
            output_words = ""
            tokens = [[self.start_token_idx]]
            yb = self.embedding(torch.LongTensor(tokens).to(self.device))
            yb = self.add_pe(yb, device=self.device)
            for i in range(max_text_len):
                #false pad masks
                yb_pad_idxs = (torch.ones(1).long()*yb.shape[1]*10000).to(self.device)
                yb_pad_masks = self.make_pad_masks(yb_pad_idxs, yb.shape[1], device=self.device)
                #false mixed pad masks
                mixed_pad_masks = self.make_pad_masks(yb_pad_idxs, yb.shape[1], xb_pad_idxs, xb.shape[1], device=self.device)
                decoded = self.decoder(yb, yb_pad_masks, encoded, mixed_pad_masks)
                out = decoded @ self.embedding.weight.t()
                asdf, idx = torch.max(out[0][i], dim=0)
                #asdf, idxs = torch.max(out[0], dim=1)
                #print(asdf[i].item())
                #idx = idxs[i].item()
                idx = idx.item()
                if idx == self.end_token_idx:
                    break
                if i == 0:
                    output_words += self.vocab[idx]
                else:
                    output_words += ' ' + self.vocab[idx]
                tokens[0].append(idx)
                #print('tokens', tokens)
                yb = self.embedding(torch.LongTensor(tokens).to(self.device))
                yb = self.add_pe(yb, device=self.device)
                #print('yb', yb)
            return output_words
    def tokenize_xb(self, sequence):
        #should return shape (1, seq_len)
        #of indices
        sequence = sequence.strip().split(' ')
        tokenized = []
        for i in range(len(sequence)):
            tokenized.append(self.vocab_hashtable[sequence[i]])
        return torch.LongTensor(tokenized).reshape(1,-1).to(self.device)

# %% [code]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.708231Z","iopub.execute_input":"2023-02-24T04:47:32.709041Z","iopub.status.idle":"2023-02-24T04:47:32.723401Z","shell.execute_reply.started":"2023-02-24T04:47:32.708994Z","shell.execute_reply":"2023-02-24T04:47:32.722209Z"}}
#from squad_dataset import Dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.725596Z","iopub.execute_input":"2023-02-24T04:47:32.726146Z","iopub.status.idle":"2023-02-24T04:47:32.734766Z","shell.execute_reply.started":"2023-02-24T04:47:32.726095Z","shell.execute_reply":"2023-02-24T04:47:32.733817Z"}}
#asdf = torch.arange(12).reshape(4,3)
#max_vals, idxs = torch.max(-1*asdf[0], dim=0)
#idxs

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.736522Z","iopub.execute_input":"2023-02-24T04:47:32.736982Z","iopub.status.idle":"2023-02-24T04:47:32.746427Z","shell.execute_reply.started":"2023-02-24T04:47:32.736938Z","shell.execute_reply":"2023-02-24T04:47:32.745369Z"}}
#ds = Dataset()
#len(ds.vocab)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.748214Z","iopub.execute_input":"2023-02-24T04:47:32.748983Z","iopub.status.idle":"2023-02-24T04:47:32.757532Z","shell.execute_reply.started":"2023-02-24T04:47:32.748934Z","shell.execute_reply":"2023-02-24T04:47:32.756219Z"}}
#my_transformer = Transformer(ds.vocab, ds.vocab_hashtable)
#result = my_transformer(torch.ones(5,2).long()*3, torch.tensor([1,2,3,4,5]).long(), torch.ones(5,7).long()*4, torch.tensor([1,2,1,3,5]).long())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.759253Z","iopub.execute_input":"2023-02-24T04:47:32.759754Z","iopub.status.idle":"2023-02-24T04:47:32.770602Z","shell.execute_reply.started":"2023-02-24T04:47:32.759701Z","shell.execute_reply":"2023-02-24T04:47:32.769431Z"}}
#my_transformer.make_inference("the bird is the word", 30)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-02-24T04:47:32.772231Z","iopub.execute_input":"2023-02-24T04:47:32.773348Z","iopub.status.idle":"2023-02-24T04:47:32.785872Z","shell.execute_reply.started":"2023-02-24T04:47:32.773297Z","shell.execute_reply":"2023-02-24T04:47:32.784743Z"}}
#result[4]

# %% [code]


# %% [code]
