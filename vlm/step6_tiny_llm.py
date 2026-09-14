"""
STEP 6 -- The LLM we are going to upgrade.
==========================================

A decoder-only language model is the SAME transformer as the ViT, with two
changes:

  1. the input is token ids -> looked up in an embedding table
  2. attention is CAUSAL: token 5 may look at 0..5 but never at 6

and at the end a Linear maps each position back to vocabulary logits:
"given everything so far, what is the next word?"

The one design decision that matters for us:  forward() accepts either
`input_ids` OR `inputs_embeds`.  Real LLMs (GPT, Llama, Qwen...) all expose
this.  It is the door through which we will smuggle the image in -- we hand
the LLM vectors that never came from the embedding table at all.
"""
import torch
import torch.nn as nn

from .step4_vit import TransformerBlock


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, dim=128, n_heads=4, n_layers=4,
                 max_seq_len=64, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)     # id -> vector
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout, causal=True)   # <-- causal!
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # weight tying: the table that turns ids into vectors also turns
        # vectors back into id-scores. Standard trick, saves parameters.
        self.lm_head.weight = self.token_embed.weight

        # Init matters more than people expect: nn.Embedding defaults to
        # N(0,1), and because lm_head shares that same matrix the logits would
        # come out ~sqrt(dim) too large and the starting loss would be huge.
        # Small init => starting loss ~= ln(vocab_size), the "no idea" baseline.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed_tokens(self, input_ids):
        """ids [B, T] -> vectors [B, T, dim].  The LLM's native 'language'."""
        return self.token_embed(input_ids)

    def forward(self, input_ids=None, inputs_embeds=None, return_attn=False):
        """
        Give it EITHER input_ids (normal text) OR inputs_embeds (our image+text
        mix).  Returns logits [B, T, vocab].
        """
        assert (input_ids is None) != (inputs_embeds is None), \
            "pass exactly one of input_ids / inputs_embeds"
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        T = x.shape[1]
        assert T <= self.pos_embed.shape[1], "sequence longer than max_seq_len"
        x = x + self.pos_embed[:, :T]            # where am I in the sequence

        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, return_attn=True)
                attns.append(a)                  # [B, heads, T, T]
            else:
                x = blk(x)
        logits = self.lm_head(self.norm(x))      # [B, T, vocab]
        return (logits, attns) if return_attn else logits


if __name__ == "__main__":
    from .tokenizer import WordTokenizer
    tok = WordTokenizer()
    llm = TinyLLM(len(tok))
    ids = torch.tensor([tok.encode("a red circle in the top left")])
    print("ids     ", tuple(ids.shape))
    print("logits  ", tuple(llm(input_ids=ids).shape), "= [B, T, vocab]")
    # and the same thing through the side door:
    emb = llm.embed_tokens(ids)
    print("via inputs_embeds:", tuple(llm(inputs_embeds=emb).shape),
          "-- identical:", torch.allclose(llm(input_ids=ids), llm(inputs_embeds=emb)))
