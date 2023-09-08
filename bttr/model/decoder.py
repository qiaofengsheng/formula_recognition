from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor
from torch.nn.modules.transformer import TransformerDecoder

from bttr.datamodule import vocab, vocab_size
from bttr.model.pos_enc import WordPosEnc, WordRotaryEmbed
from bttr.utils import Hypothesis, to_tgt_output


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerDecoder:
    """build transformer decoder with params
    Parameters
    ----------
    d_model : int
    nhead : int
    num_decoder_layers : int
    dim_feedforward : int
    dropout : float
    Returns
    -------
    nn.TransformerDecoder
    """
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    return decoder


class Decoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, t, d]
        src_mask: LongTensor
            [b, t]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)    # att_mask 是一个上三角矩阵，右上角为1
        tgt_pad_mask = tgt == vocab.PAD_IDX # 获取标签的padding mask

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]

        src = rearrange(src, "b t d -> t b d")  # 预测值 batch,seq_len,d_model -> seq_len,batch,d_model
        tgt = rearrange(tgt, "b l d -> l b d")  # 标签值 batch,seq_len,d_model -> seq_len,batch,d_model

        out = self.model(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,      
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )   # pytorch中的mask都是将要保留的位置变为false，mask的位置为True

        out = rearrange(out, "l b d -> b l d")  # 预测值 seq_len,batch,d_model -> batch,seq_len,d_model
        out = self.proj(out)

        return out

    def _beam_search(
        self,
        src: FloatTensor,
        mask: LongTensor,
        direction: str,
        beam_size: int,
        max_len: int,
    ) -> List[Hypothesis]:  # https://deconx.cn/blog/2022/09/01/beam-search
        """run beam search for one direction

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        direction : str
            one of "l2r" and "r2l"
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        assert direction in {"l2r", "r2l"}
        assert (
            src.size(0) == 1 and mask.size(0) == 1
        ), f"beam search should only have single source, encounter with batch_size: {src.size(0)}"

        if direction == "l2r":
            start_w = vocab.SOS_IDX
            stop_w = vocab.EOS_IDX
        else:
            start_w = vocab.EOS_IDX
            stop_w = vocab.SOS_IDX

        hypotheses = torch.full(
            (1, max_len + 1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=self.device,
        )   # 创建一个201长度的张量
        hypotheses[:, 0] = start_w  # 初始位置设置为开始标志符号的对应索引

        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)  # 置信度
        completed_hypotheses: List[Hypothesis] = [] # 存放已经计算完毕的序列

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
            exp_mask = repeat(mask.squeeze(0), "s -> b s", b=hyp_num)

            decode_outputs = self(exp_src, exp_mask, hypotheses)[:, t, :]   # 获取当前解码的对应序列位置的结果
            log_p_t = F.log_softmax(decode_outputs, dim=-1)     # softmax

            live_hyp_num = beam_size - len(completed_hypotheses)    # 计算剩余需要的序列数量
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e=vocab_size)
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                continuous_hyp_scores, k=live_hyp_num
            )       # 获取topk的置信度和对应的索引位置

            prev_hyp_ids = top_cand_hyp_pos // vocab_size   # 找到之后，怎么确定这前 k 个最大的是哪个序列，以及选择的词表中的哪个词呢？由于 contiuating_hyp_scores: (hyp_num * vocab_size,), 故作商就得到了具体的序列，余数即为对应词表的词，太秒了！！
            hyp_word_ids = top_cand_hyp_pos % vocab_size    # 获取对应的单词index

            t += 1
            new_hypotheses = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                hypotheses[prev_hyp_id, t] = hyp_word_id

                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(
                            seq_tensor=hypotheses[prev_hyp_id, 1:t]
                            .detach()
                            .clone(),  # remove START_W at first
                            score=cand_new_hyp_score,
                            direction=direction,
                        )
                    )
                else:
                    new_hypotheses.append(hypotheses[prev_hyp_id].detach().clone())
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = torch.stack(new_hypotheses, dim=0)
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=hypotheses[0, 1:].detach().clone(),
                    score=hyp_scores[0].detach().item(),
                    direction=direction,
                )
            )

        return completed_hypotheses

    def _cross_rate_score(
        self,
        src: FloatTensor,
        mask: LongTensor,
        hypotheses: List[Hypothesis],
        direction: str,
    ) -> None:
        """give hypotheses to another model, add score to hypotheses inplace

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask : LongTensor
            [1, l]
        hypotheses : List[Hypothesis]
        direction : str
        """
        assert direction in {"l2r", "r2l"}
        indices = [h.seq for h in hypotheses]
        tgt, output = to_tgt_output(indices, direction, self.device)

        b = tgt.size(0)
        exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
        exp_mask = repeat(mask.squeeze(0), "s -> b s", b=b)

        output_hat = self(exp_src, exp_mask, tgt)

        flat_hat = rearrange(output_hat, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        loss = F.cross_entropy(
            flat_hat, flat, ignore_index=vocab.PAD_IDX, reduction="none"
        )

        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)

        for i, l in enumerate(loss):
            score = -l
            hypotheses[i].score += score

    def beam_search(
        self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run beam search for src img

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        l2r_hypos = self._beam_search(src, mask, "l2r", beam_size, max_len) # 从左到右进行decoder预测
        self._cross_rate_score(src, mask, l2r_hypos, direction="r2l")   

        r2l_hypos = self._beam_search(src, mask, "r2l", beam_size, max_len) # 从右到左进行decoder预测
        self._cross_rate_score(src, mask, r2l_hypos, direction="l2r")
        return l2r_hypos + r2l_hypos
