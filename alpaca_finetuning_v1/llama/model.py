# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int = 10
    adapter_layer: int = 20


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, adapter_len + slen)
        if adapter is not None:
            scores = torch.cat(
                [
                    self.gate.tanh().half() * F.softmax(scores[:, :, :, :adapter_len].float(), dim=-1).type_as(xq),
                    F.softmax(scores[:, :, :, adapter_len:].float(), dim=-1).type_as(xq),
                ],
                dim=-1,
            )
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # when no adapter, scores dim (bs, n_local_heads, slen, slen), values dim (bs, n_local_heads, slen, head_dim)
        # when with adapter, scores dim (bs, n_local_heads, slen, adapter_len+slen), values dim (bs, n_local_heads, adapter_len+slen, head_dim)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # concat all attention head

        return self.wo(output)


    def forward_with_adap(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, adapter_len + slen)
        if adapter is not None:
            scores = torch.cat(
                [
                    self.gate.tanh().half() * F.softmax(scores[:, :, :, :adapter_len].float(), dim=-1).type_as(xq),
                    F.softmax(scores[:, :, :, adapter_len:].float(), dim=-1).type_as(xq),
                ],
                dim=-1,
            )
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # when no adapter, scores dim (bs, n_local_heads, slen, slen), values dim (bs, n_local_heads, slen, head_dim)
        # when with adapter, scores dim (bs, n_local_heads, slen, adapter_len+slen), values dim (bs, n_local_heads, adapter_len+slen, head_dim)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # concat all attention head

        # this is representation created by adapter

        if adapter is not None:
            # print("==== scores[:, :, :, :5]", scores[:, :, :, :5].shape)
            # print("==== values[:, :, :5, :]", values[:, :, :5, :].shape)
            output_adap = torch.matmul(scores[:, :, :, :1], values[:, :, :1, :])  # (bs, n_local_heads, slen, head_dim)
            output_adap = output_adap.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # concat all attention head
            # print("==== output_adap", output_adap.shape)
            return self.wo(output), self.wo(output_adap)
        else:
            # print("No adapter!")
            return self.wo(output), None






class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TransformerBlockCase2(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):
        h, h_adap = self.attention.forward_with_adap(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, h_adap


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    def forward(self, examples, labels):

        _bsz, seqlen = examples.shape

        # print("==== examples", examples)
        # print("==== labels", labels)

        with torch.no_grad():
            h = self.tok_embeddings(examples) # shape [batch_size, max_seq_len, 4096]
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = None
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
            start_pos = 0
            # layers without adapter
            for layer in self.layers[: -1 * self.adapter_layer]:
                h = layer(h, start_pos, freqs_cis, mask)

        adapter_index = 0
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, 4096).unsqueeze(1)
        # layers with adapter
        for layer in self.layers[-1 * self.adapter_layer :]:
            h = layer(h, start_pos, freqs_cis, mask, adapter[adapter_index].half())
            adapter_index = adapter_index + 1


        h = self.norm(h) # shape [batch_size, max_seq_len, 4096]
        output = self.output(h) # shape [batch_size, max_seq_len, vocab_size]
        output = output[:, :-1, :].reshape(-1, self.vocab_size) # shape [batch_size * max_seq_len, vocab_size]
        labels = labels[:, 1:].flatten() # shape [batch_size * max_seq_len]

        c_loss = self.criterion(output, labels)

        return c_loss



class TransformerContra(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.recon_loss = torch.nn.MSELoss(reduction='sum')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    def forward(self, examples1, labels1, examples2, labels2, epoch):

        assert examples1.shape == examples2.shape

        _bsz, seqlen = examples1.shape

        # print("==== For example 1")
        with torch.no_grad():
            h1 = self.tok_embeddings(examples1)  # shape [batch_size, max_seq_len, 4096]
            freqs_cis = self.freqs_cis.to(h1.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = None
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h1.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h1)
            start_pos = 0
            # layers without adapter
            for layer in self.layers[: -1 * self.adapter_layer]:
                h1 = layer(h1, start_pos, freqs_cis, mask)

        adapter_index = 0
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, 4096).unsqueeze(1)
        # layers with adapter
        # h_adapter1 = []
        for layer in self.layers[-1 * self.adapter_layer:]:
            h1 = layer(h1, start_pos, freqs_cis, mask, adapter[adapter_index].half())
            # h_adapter1.append(h1)
            adapter_index = adapter_index + 1
            # print("==== layers 2", h1.shape)

        h1 = self.norm(h1)  # shape [batch_size, max_seq_len, 4096]
        output1 = self.output(h1)  # shape [batch_size, max_seq_len, vocab_size]
        output1 = output1[:, :-1, :].reshape(-1, self.vocab_size)  # shape [batch_size * max_seq_len, vocab_size]
        labels1 = labels1[:, 1:].flatten()  # shape [batch_size * max_seq_len]

        c_loss1 = self.criterion(output1, labels1)


        # print("==== For example 2")
        with torch.no_grad():
            h2 = self.tok_embeddings(examples2)  # shape [batch_size, max_seq_len, 4096]
            freqs_cis = self.freqs_cis.to(h2.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = None
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h2.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h2)
            start_pos = 0
            # layers without adapter
            for layer in self.layers[: -1 * self.adapter_layer]:
                h2 = layer(h2, start_pos, freqs_cis, mask)

        adapter_index = 0
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, 4096).unsqueeze(1)
        # layers with adapter
        # h_adapter2 = []
        for layer in self.layers[-1 * self.adapter_layer:]:
            h2 = layer(h2, start_pos, freqs_cis, mask, adapter[adapter_index].half())
            # h_adapter2.append(h2)
            adapter_index = adapter_index + 1
            # print("==== layers 2", h2.shape)

        h2 = self.norm(h2)  # shape [batch_size, max_seq_len, 4096]
        output2 = self.output(h2)  # shape [batch_size, max_seq_len, vocab_size]
        output2 = output2[:, :-1, :].reshape(-1, self.vocab_size)  # shape [batch_size * max_seq_len, vocab_size]
        labels2 = labels2[:, 1:].flatten()  # shape [batch_size * max_seq_len]

        c_loss2 = self.criterion(output2, labels2)
        print("===== c_loss1, c_loss2", c_loss1.item(), c_loss2.item())

        ########################
        # Add contrastive Loss
        ########################

        # h_adapter1 = torch.stack(h_adapter1[-1:]) # only take last layer for now
        # h_adapter2 = torch.stack(h_adapter2[-1:])

        # torch.set_printoptions(profile="full")
        # # print("==== h1", h1.shape, h1)
        # # print("==== h2", h2.shape, h2)
        # self_cal_mse = (h1/10 - h2/10) ** 2
        # # print("==== self_cal_mse", self_cal_mse.shape, self_cal_mse)
        # print("==== self_cal_mse sum", self_cal_mse.sum())
        # print("==== self_cal_mse max", self_cal_mse.max())
        # print("==== self_cal_mse min", self_cal_mse.min())


        # print("==== h_adapter1", h_adapter1.shape, h_adapter1)
        # print("==== h_adapter2", h_adapter2.shape, h_adapter2)
        # self_cal_mse = (h_adapter1-h_adapter2)**2
        # print("==== self_cal_mse", self_cal_mse.shape, self_cal_mse)
        # self_cal_mse = self_cal_mse.sum(dim=1)
        # print("==== self_cal_mse sum", self_cal_mse)

        # If normalize
        # h_adapter1 = F.normalize(h_adapter1, dim=1)
        # h_adapter2 = F.normalize(h_adapter2, dim=1)
        # print("==== h_adapter1 norm", h_adapter1.shape, h_adapter1)
        # print("==== h_adapter2 norm", h_adapter2.shape, h_adapter2)

        contra_loss = self.recon_loss(h1/100, h2/100)
        print("==== 0.1 * contra_loss", 0.1 * contra_loss.item())
        total_loss = c_loss1 + c_loss2 + contra_loss * 0.1
        print("==== total loss", total_loss)

        return total_loss





class TransformerContraCase2(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.recon_loss = torch.nn.MSELoss()

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlockCase2(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    def forward(self, examples, labels):

        _bsz, seqlen = examples.shape

        # print("==== examples", examples)
        # print("==== labels", labels)

        with torch.no_grad():
            h = self.tok_embeddings(examples) # shape [batch_size, max_seq_len, 4096]
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = None
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
            start_pos = 0
            # layers without adapter
            for layer in self.layers[: -1 * self.adapter_layer]:
                h, _ = layer(h, start_pos, freqs_cis, mask)

        adapter_index = 0
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, 4096).unsqueeze(1)
        # layers with adapter
        h_adaps = []
        for layer in self.layers[-1 * self.adapter_layer :]:
            h, h_adap = layer(h, start_pos, freqs_cis, mask, adapter[adapter_index].half())
            h_adaps.append(h_adap)
            # print("==== append h_adap", h_adap.shape)
            adapter_index = adapter_index + 1

        h = self.norm(h) # shape [batch_size, max_seq_len, 4096]
        output = self.output(h) # shape [batch_size, max_seq_len, vocab_size]
        output = output[:, :-1, :].reshape(-1, self.vocab_size) # shape [batch_size * max_seq_len, vocab_size]
        labels = labels[:, 1:].flatten() # shape [batch_size * max_seq_len]

        c_loss = self.criterion(output, labels)

        # Add contrastive loss
        # print("==== len(h_adaps)", len(h_adaps))
        # contra_loss = self.recon_loss(h_adaps[0][0, :, :]*100, h_adaps[0][1, :, :]*100)
        # contra_loss = self.recon_loss(h_adaps[0][0, :, :], h_adaps[0][1, :, :])


        h_adaps = torch.stack(h_adaps[-1:], dim=1)
        # contra_loss = self.recon_loss(h_adaps[0, :, :, :], h_adaps[1, :, :, :])
        # print("==== h_adaps", h_adaps.shape) # bsz * adap_layer * seq_len * 4096

        # contra_loss = torch.var(h_adaps, dim=0).sum()
        contra_loss = torch.var(h_adaps, dim=0).mean()

        print("==== contra_loss", contra_loss.item())
        print("==== c_loss", c_loss.item())

        total_loss = c_loss + 0 * contra_loss

        return total_loss



