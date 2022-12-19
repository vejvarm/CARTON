import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import constants
from constants import *
from args import parse_and_get_args
args = parse_and_get_args()


class CARTON(nn.Module):
    def __init__(self, vocabs):
        super(CARTON, self).__init__()
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs[INPUT], DEVICE)
        self.decoder = Decoder(vocabs[LOGICAL_FORM], DEVICE)
        self.ner = NerNet(len(vocabs[NER]))
        self.coref = CorefNet(len(vocabs[COREF]))
        self.stptr_net = StackedPointerNetworks(vocabs[PREDICATE_POINTER], vocabs[TYPE_POINTER])

    def forward(self, src_tokens, trg_tokens):
        encoder_out = self.encoder(src_tokens)
        ner_out, ner_h = self.ner(encoder_out)  # ANCHOR: LASAGNE
        coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))  # ANCHOR: LASAGNE
        decoder_out, decoder_h = self.decoder(src_tokens, trg_tokens, encoder_out)
        encoder_ctx = encoder_out[:, -1:, :]  # ANCHOR [batch_size, time, encoder_dim]
        stacked_pointer_out = self.stptr_net(encoder_ctx, decoder_h)  # ANCHOR encoder context vector

        return {
            LOGICAL_FORM: decoder_out,
            PREDICATE_POINTER: stacked_pointer_out[PREDICATE_POINTER],  # (bs, lf_actions*n_predicates)
            TYPE_POINTER: stacked_pointer_out[TYPE_POINTER],     # (bs, lf_actions*n_types)
            NER: ner_out,
            COREF: coref_out
        }

    # ANCHOR: LASAGNE
    def _predict_encoder(self, src_tensor):
        with torch.no_grad():
            encoder_out = self.encoder(src_tensor)
            ner_out, ner_h = self.ner(encoder_out)
            coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))

        return {
            ENCODER_OUT: encoder_out,
            NER: ner_out,
            COREF: coref_out
        }

    def _predict_decoder(self, src_tokens, trg_tokens, encoder_out):
        with torch.no_grad():
            decoder_out, decoder_h = self.decoder(src_tokens, trg_tokens, encoder_out)
            encoder_ctx = encoder_out[:, -1:, :]

            return {
                DECODER_OUT: decoder_out,
                DECODER_H: decoder_h,
            }


class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(-1, x.shape[-1])


# ANCHOR: BIO entity labeling (annotator)
#   Finding all entities in the input utterance (BIO & types)
class NerNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(NerNet, self).__init__()
        self.ner_lstm = nn.Sequential(
            nn.LSTM(input_size=args.emb_dim, hidden_size=args.emb_dim, batch_first=True),
            LstmFlatten(),
            nn.LeakyReLU()
        )

        self.ner_linear = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )

    def forward(self, x):
        h = self.ner_lstm(x)
        return self.ner_linear(h), h


# ANCHOR: This is interesting
class CorefNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(CorefNet, self).__init__()
        self.seq_net = nn.Sequential(
            nn.Linear(args.emb_dim * 2, args.emb_dim),
            nn.LeakyReLU(),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )

    def forward(self, x):
        return self.seq_net(x)


class PointerStack(nn.Module):
    def __init__(self, vocab):
        super(PointerStack, self).__init__()
        self.kg_items = torch.tensor(list(vocab.stoi.values())).to(DEVICE)
        self.embeddings = nn.Embedding(len(vocab), args.emb_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.tahn = nn.Tanh()
        self.flatten = Flatten()
        hidden = []
        # create ptr_n_hidden Linear+ReLU layers with halving sizes of emb_dim
        for i in range(args.ptr_n_hidden):
            hidden.append(nn.Linear(args.emb_dim // (2 ** i), args.emb_dim // (2 ** (i + 1))))
            hidden.append(nn.ReLU())
        self.lin_hidden = nn.Sequential(*hidden)
        self.linear_out = nn.Linear(args.emb_dim//(2**args.ptr_n_hidden), 1)

    def forward(self, x):  # ANCHOR Pointer network
        # x.shape: [25, n, 1, 300] ... inputed from ANCHOR@StackedPointerNetworks forward function
        embed = self.embeddings(self.kg_items).unsqueeze(0)
        # print(f"before: {x.shape}") # torch.Size([25, 18, 1, 300])
        x = x.expand(x.shape[0], x.shape[1], embed.shape[1], x.shape[-1])
        # print(f"after: {x.shape}") # torch.Size([25, 18, 1560, 300])
        x = x + embed.expand(x.shape[0], x.shape[1], embed.shape[1], embed.shape[-1])
        # print(f"forever_after: {x.shape}")  # torch.Size([25, 18, 1560, 300])
        x = self.tahn(x)
        x = self.lin_hidden(x)
        x = self.linear_out(x)
        # print(f"after linear: {x.shape}")  # torch.Size([25, 18, 1560, 1])
        x = x.squeeze(-1)
        # print(f"after squeeze: {x.shape}")  # torch.Size([25, 18, 1560])
        x = self.flatten(x)
        # print(f"after flatten: {x.shape}")  # torch.Size([450, 1560]) !!! torch.Flatten class is overridden

        return x


class StackedPointerNetworks(nn.Module):
    def __init__(self, predicate_vocab, type_vocab):
        super(StackedPointerNetworks, self).__init__()

        self.context_linear = nn.Linear(args.emb_dim*2, args.emb_dim)
        self.dropout = nn.Dropout(args.dropout)

        self.predicate_pointer = PointerStack(predicate_vocab)
        self.type_pointer = PointerStack(type_vocab)

    def forward(self, encoder_ctx, decoder_h):
        x = torch.cat([encoder_ctx.expand(decoder_h.shape), decoder_h], dim=-1)  # ANCHOR: this is gonna be problematic!
        # TODO: each entry in decoder_h is concatenated by encoder_h
        #  e.g. expand([25, 1, 300], dim=1, n) concat with [25, n, 300] => [25, n, 600]
        x = self.context_linear(x).unsqueeze(2)  # [25, n, 600] => [25, n, 1, 300]
        x = self.dropout(x)

        return {
            PREDICATE_POINTER: self.predicate_pointer(x),
            TYPE_POINTER: self.type_pointer(x),
        }


class Encoder(nn.Module):
    def __init__(self, vocabulary, device, embed_dim_out=args.emb_dim, layers=args.layers,
                 heads=args.heads, pf_dim=args.pf_dim, dropout=args.dropout, max_positions=args.max_positions):
        super().__init__()
        input_dim = len(vocabulary)
        self.padding_idx = vocabulary.stoi[PAD_TOKEN]
        self.dropout = dropout
        self.device = device

        input_dim, embed_dim = vocabulary.vectors.size()
        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(input_dim, embed_dim)
        self.embed_tokens.weight.data.copy_(vocabulary.vectors)
        self.embed_positions = PositionalEmbedding(embed_dim, dropout, max_positions)

        # Feed-Forward layer to transform fixed Emb vector dimensions
        self.ff_emb = nn.Linear(embed_dim, embed_dim_out, True, device)  # ANCHOR EMBDIM: if you want args.emb_dim different than 300, uncomment this

        # Stack Encoder Transformer Layers
        self.layers = nn.ModuleList([EncoderLayer(embed_dim_out, heads, pf_dim, dropout, device) for _ in range(layers)])

    def forward(self, src_tokens):
        src_mask = (src_tokens != self.padding_idx).unsqueeze(1).unsqueeze(2)

        x = self.embed_tokens(src_tokens) * self.scale
        x += self.embed_positions(src_tokens)
        # print(x.shape) # torch.Size([batch size, utterance length, emb_dim_size]) torch.Size([25, 43, 300])
        # x = self.ff_emb(x)  # ANCHOR EMBDIM: if you want args.emb_dim different than 300, uncomment this
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, src_mask):
        x = self.layer_norm(src_tokens + self.dropout(self.self_attn(src_tokens, src_tokens, src_tokens, src_mask)))  # ANCHOR: Encoder takes whole encoder_output (not only h_ctx)
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, vocabulary, device, embed_dim=args.emb_dim, layers=args.layers,
                 heads=args.heads, pf_dim=args.pf_dim, dropout=args.dropout, max_positions=args.max_positions):
        super().__init__()

        output_dim = len(vocabulary)
        self.pad_id = vocabulary.stoi[PAD_TOKEN]
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.max_positions = max_positions

        self.scale = math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(output_dim, embed_dim)
        self.embed_positions = PositionalEmbedding(embed_dim, dropout, max_positions)

        self.layers = nn.ModuleList([DecoderLayer(embed_dim, heads, pf_dim, dropout, device) for _ in range(layers)])

        self.linear_out = nn.Linear(embed_dim, output_dim)

    def make_masks(self, src_tokens, trg_tokens):
        src_mask = (src_tokens != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg_tokens != self.pad_id).unsqueeze(1).unsqueeze(3)
        trg_len = trg_tokens.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return src_mask, trg_mask

    def forward(self, src_tokens, trg_tokens, encoder_out):
        src_mask, trg_mask = self.make_masks(src_tokens, trg_tokens)

        x = self.embed_tokens(trg_tokens) * self.scale
        x += self.embed_positions(trg_tokens)
        h = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            h = layer(h, encoder_out, trg_mask, src_mask)

        x = h.contiguous().view(-1, h.shape[-1])
        x = self.linear_out(x)

        return x, h


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.src_attn = MultiHeadedAttention(embed_dim, heads, dropout, device)
        self.pos_ff = PositionwiseFeedforward(embed_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embed_trg, embed_src, trg_mask, src_mask):
        x = self.layer_norm(embed_trg + self.dropout(self.self_attn(embed_trg, embed_trg, embed_trg, trg_mask)))
        x = self.layer_norm(x + self.dropout(self.src_attn(x, embed_src, embed_src, src_mask)))
        x = self.layer_norm(x + self.dropout(self.pos_ff(x)))

        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout, device):
        super().__init__()
        # print(embed_dim)
        # print(heads)
        assert embed_dim % heads == 0
        self.attn_dim = embed_dim // heads
        self.heads = heads
        self.dropout = dropout

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.attn_dim])).to(device)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        Q = Q.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)
        K = K.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)
        V = V.view(batch_size, -1, self.heads, self.attn_dim).permute(0, 2, 1, 3) # (batch, heads, sent_len, attn_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # (batch, heads, sent_len, sent_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1) # (batch, heads, sent_len, sent_len)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        x = torch.matmul(attention, V) # (batch, heads, sent_len, attn_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # (batch, sent_len, heads, attn_dim)
        x = x.view(batch_size, -1, self.heads * (self.attn_dim)) # (batch, sent_len, embed_dim)
        x = self.linear_out(x)

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_dim, pf_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, pf_dim)
        self.linear_2 = nn.Linear(pf_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.linear_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        pos_embed = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        return Variable(self.pos_embed[:, :x.size(1)], requires_grad=False)
