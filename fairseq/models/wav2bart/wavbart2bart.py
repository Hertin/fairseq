# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Optional, Any
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer


@dataclass
class WavBart2BartConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None

    fix_extractor: bool = False

    autoregressive: bool = II("task.autoregressive")

    bart_path: str = field(
        default="",
        metadata={"help": "path of bart model"},
    )

    fix_encoder: bool = False
    fix_decoder: bool = False
    fix_bart_encoder: bool = True

    pad_token: int = field(
        default=1, metadata={"help": "pad token"}
    )

    mix_normalization_factor: float = field(
        default=100000, metadata={"help": "mix_normalization_factor"}
    )


@register_model("wavbart2bart", dataclass=WavBart2BartConfig)
class WavBart2Bart(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: WavBart2BartConfig, task: FairseqTask):
        """Build a new model instance."""
        from fairseq.models.bart import BARTModel
        if os.path.isfile(os.path.join(cfg.bart_path, 'model.pt')):
            print('loading bart from cfg path')
            bart = BARTModel.from_pretrained(cfg.bart_path, checkpoint_file='model.pt')
        else:
            print('loading bart from relative path')
            bart = BARTModel.from_pretrained('models/bart.base', checkpoint_file='model.pt')

        assert cfg.autoregressive, "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder = cls.build_encoder(cfg, bart)
        decoder = cls.build_decoder(cfg, bart)
        model = WavBart2Bart(encoder, decoder)
        return model

    @classmethod
    def build_encoder(cls, cfg: WavBart2BartConfig, bart=None):
        encoder = Wav2VecEncoder(cfg, bart=bart)
        if cfg.fix_encoder:
            print('fix w2v encoder')
            for parameter in encoder.parameters():
                parameter.requires_grad = False

        return encoder

    @classmethod
    def build_decoder(cls, cfg: WavBart2BartConfig, bart=None):
        decoder = BartDecoder(cfg, bart=bart)
        if cfg.fix_decoder:
            for n, parameter in decoder.named_parameters():
                if 'decoder.embed_positions' in n or 'decoder.embed_tokens' in n:
                    continue
                parameter.requires_grad = False

        return decoder

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=True, **kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: WavBart2BartConfig, tgt_dict=None, bart=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            if os.path.isfile(os.path.join(cfg.w2v_path)):
                print('load wav2vec from cfg path')
                state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            else:
                print('load wav2vec from relative path')
                state = checkpoint_utils.load_checkpoint_to_cpu('models/wav2vec_small.pt', arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.bart_encoder = bart.model.encoder
        bart_encoder = bart.model.encoder
        self.bart_encoder = TransformerEncoder(bart_encoder.args, bart_encoder.dictionary, bart_encoder.embed_tokens)
        self.bart_encoder.load_state_dict(bart_encoder.state_dict())
        self.fix_bart_encoder = cfg.fix_bart_encoder

        if self.fix_bart_encoder:
            print('fix bart encoder')
            for n, parameter in self.bart_encoder.named_parameters():
                parameter.requires_grad = False

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

        self.pad_token = cfg.pad_token
        self.mix_normalization_factor = cfg.mix_normalization_factor

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        input_lengths = (1 - padding_mask.long()).sum(-1)
        output_length = torch.max(self.w2v_model._get_feat_extract_output_lengths(input_lengths))
        # print('output_lengths', output_length,  'self.pad_token', self.pad_token)
        # print('kwargs', kwargs['bart_input_tokens'].shape, kwargs['bart_input_tokens'].type())
        batch_size, ntoken = kwargs['bart_input_tokens'].shape
        bart_input = torch.zeros(batch_size, output_length).long().fill_(self.pad_token).to(kwargs['bart_input_tokens'])
        bart_input[:, :ntoken] = kwargs['bart_input_tokens']
        # print(bart_input, bart_input.shape)
        # raise
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        x_bart = self.bart_encoder(
            src_tokens=bart_input,
            src_lengths=None,
            token_embeddings=None,
            return_all_hiddens=False
        )

        if self.proj:
            x = self.proj(x)
        x_bart = x_bart['encoder_out'][0]
        # print('x.shape', x.shape, )
        # print('x_bart', x_bart['encoder_out'][0].shape)
        # print(x_bart['encoder_padding_mask'][0].shape)
        prob = torch.sigmoid(torch.FloatTensor(
            [self.num_updates / self.mix_normalization_factor]
        )) * 2 - 1
        # n_mix = int(self.mix_rate * output_length)
        # indices = torch.randperm(output_length)[:n_mix]
        # print(n_mix, indices)
        # print(prob)
        # mask = torch.bernoulli(torch.full(x.shape, prob.item())).int().to(x)
        mask = torch.bernoulli(torch.full(x.shape[:1], prob.item()))[:,None,None].to(x)
        reverse_mask = 1 - mask
        x = x * mask + x_bart * reverse_mask
        # x_bart[indices,:,:] = x[indices,:,:]

        # print('self.num_updates', prob, self.num_updates)
        if self.num_updates % 1000 == 0:
            print('self.num_updates', prob, self.num_updates)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)] # T x B x C

        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class BartDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: WavBart2BartConfig,
        dictionary=None,
        embed_tokens=None,
        no_encoder_attn=False,
        bart=None
    ):
        super().__init__(dictionary)
        self.cfg = cfg
        # bart = torch.hub.load('pytorch/fairseq', 'bart.base')
        bart_decoder = bart.model.decoder
        self.decoder = TransformerDecoder(bart_decoder.args, bart_decoder.dictionary, bart_decoder.embed_tokens)
        self.decoder.load_state_dict(bart_decoder.state_dict())

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # with torch.no_grad() if self.cfg.fix_decoder else contextlib.ExitStack():
        x, extra = self.decoder(prev_output_tokens, encoder_out, incremental_state)

        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        self.decoder.extract_features(prev_output_tokens, encoder_out, incremental_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decoder.max_positions()

    def buffered_future_mask(self, tensor):
        
        return self.decoder.buffered_future_mask

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
