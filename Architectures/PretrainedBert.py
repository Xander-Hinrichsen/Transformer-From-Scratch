import torch.nn as nn
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
class Transformer(nn.Module):
    def __init__(self,freeze_embeddings=True,device=device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        embedding_weights = model.embeddings.word_embeddings.weight.data
        self.embedding = torch.nn.Embedding(embedding_weights.shape[0], embedding_weights.shape[1], padding_idx=0)
        self.embedding.weight = nn.Parameter(embedding_weights)
        with torch.no_grad():
            self.embedding.weight[0] = torch.zeros(embedding_weights.shape[1])
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        self.d_model = 768
        self.transformer = torch.nn.Transformer(d_model=self.d_model, nhead=12, batch_first=True)
        self.device=device
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
        def padded_pe(xb, tokens, device=device):
            pe = positional_encoding(xb.shape[1], xb.shape[2], device=device)
            mask = (tokens == 0).reshape(xb.shape[0], -1, 1).to(device)
            return pe * mask
        self.padded_pe = padded_pe
        self.start_token = 101
        self.end_token = 102
    def forward(self,src, in_targs):
        src_key_padding_mask = src == 0
        tgt_key_padding_mask = in_targs == 0
        src_embeddings = self.embedding(src)
        targs_embeddings = self.embedding(in_targs)
        src_embeddings_pe = src_embeddings + self.padded_pe(src_embeddings, src)
        targs_embeddings_pe = targs_embeddings + self.padded_pe(targs_embeddings, in_targs)
        preds = self.transformer(src_embeddings_pe, targs_embeddings_pe,
                              tgt_mask=self.transformer.generate_square_subsequent_mask(in_targs[0].shape[0]).to(device),
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)
        #final linear layer
        preds = preds @ self.embedding.weight.t()
        
        #mask before final softmax to calculate the loss
        def make_result_masks(preds, targs, device=self.device):
            mask = torch.ones_like(preds).to(device) * ((targs == 0).reshape(preds.shape[0],-1,1)).to(device)
            mask[mask==1] = -torch.inf
            pad_indices = mask[:,:,0] == 0
            mask[:,:,0] = 0
            mask[:,:,0][pad_indices] = -torch.inf
            return mask
        #mask before final softmax to calculate the loss
        preds = preds + make_result_masks(preds, in_targs)
        return preds
    def make_inference(self,txt, max_len):
        src = torch.LongTensor(self.tokenizer.encode(txt)[1:-1]).reshape(1,-1).to(self.device)
        tokens = [self.start_token]
        in_targ = torch.LongTensor([tokens]).to(self.device)
        for i in range(max_len):
            preds = self.forward(src, in_targ)
            asdf, idx = torch.max(preds[0][i], dim=0)
            idx = idx.item()
            if idx == self.end_token:
                    break
            tokens.append(idx)
            in_targ = torch.LongTensor([tokens]).to(self.device)
        return self.tokenizer.decode(tokens[1:])