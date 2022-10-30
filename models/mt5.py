import copy
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any, List

import torch
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.models.mt5 import MT5ForConditionalGeneration
from transformers.models.mt5 import MT5Tokenizer
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Stack,
    Seq2SeqLMOutput, T5ForConditionalGeneration, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, T5Block, T5LayerNorm
)

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("MT5")


@dataclass
class Seq2SeqLMPredictionOutput(Seq2SeqLMOutput):
    generated_seq: List[str] = None
    cls_logits: torch.FloatTensor = None


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class MT5ForSeq2SeqAndSeqClassification(MT5ForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: T5Config, tokenizer: str, gradient_checkpointing: bool = False):
        config.gradient_checkpointing = gradient_checkpointing
        super().__init__(config)
        self.config = config
        # self.config.gradient_checkpointing = gradient_checkpointing
        self.num_labels = getattr(config, "num_labels", 0)
        if self.num_labels > 0:
            self.classification_head = BartClassificationHead(
                input_dim=config.d_model,
                inner_dim=config.d_ff,
                num_classes=self.num_labels,
                pooler_dropout=config.dropout_rate,
            )
            self.encoder._init_weights(self.classification_head.dense)
            self.encoder._init_weights(self.classification_head.out_proj)
            self.cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.classification_head = None
            self.cls_loss_fct = None

        self.init_metric("loss", "acc", "cls_acc", "cls_loss")

        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer, use_fast=False)

        self.generating = False

    def generate(self, *model_args, **model_kwargs):
        self.generating = True
        res = super().generate(*model_args, **model_kwargs)
        self.generating = False
        return res

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cls_labels: Optional[torch.LongTensor] = None,
            meta_data: Dict[str, Any] = None,
            disable_decoder: bool = False,
            **kwargs
    ) -> Union[Dict[str, Tensor], T5Stack]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> import torch

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", return_tensors="pt")).unsqueeze(0)  # Batch size 1
        >>> outputs = model.generate(input_ids)

        >>> print("Generated: {}".format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        Generated: Hello, my dog is cute
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.num_labels > 0 and not self.generating:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.classification_head = self.classification_head.to(self.encoder.first_device)
                sentence_representation = sentence_representation.to(self.classification_head.dense.weight.device)

            logits = self.classification_head(sentence_representation)
        else:
            logits = None

        if disable_decoder:
            return Seq2SeqLMPredictionOutput(
                cls_logits=logits if logits is not None else None,
            )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        eval_gen_sentences = None
        if labels is not None:
            label_padding_mask = labels == self.config.pad_token_id
            labels[label_padding_mask] = -1
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom):
            #  Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if not self.training:
                # Generate sentences for BLEU evaluation
                max_output_length = labels.size(1)
                # Greedy decoding.
                eval_gen_sentences = self.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_output_length,
                                                   num_beams=1, do_sample=False)
                eval_gen_sentences = self.tokenizer.batch_decode(eval_gen_sentences, skip_special_tokens=True)

                acc, true_label_num = layers.get_accuracy(lm_logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if cls_labels is not None and self.num_labels > 0:
            cls_loss = self.cls_loss_fct(logits, cls_labels)

            loss = loss + cls_loss if loss is not None else cls_loss

            if not self.training:
                cls_acc, cls_true_label_num = layers.get_accuracy(logits, cls_labels)
                self.eval_metrics.update("cls_acc", cls_acc, n=cls_true_label_num)
                self.eval_metrics.update("cls_loss", cls_loss.item(), n=cls_true_label_num)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMPredictionOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            generated_seq=eval_gen_sentences,
            cls_logits=logits if logits is not None else None,
        )


class MT5ForSeq2SeqAndSeqClassificationGrounding(MT5ForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: T5Config, tokenizer: str, gradient_checkpointing: bool = False):
        config.gradient_checkpointing = gradient_checkpointing
        super().__init__(config)
        self.config = config
        # self.config.gradient_checkpointing = gradient_checkpointing
        self.num_labels = getattr(config, "num_labels", 0)
        if self.num_labels > 0:
            self.classification_head = BartClassificationHead(
                input_dim=config.d_model,
                inner_dim=config.d_ff,
                num_classes=self.num_labels,
                pooler_dropout=config.dropout_rate,
            )
            self.encoder._init_weights(self.classification_head.dense)
            self.encoder._init_weights(self.classification_head.out_proj)
            self.cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.classification_head = None
            self.cls_loss_fct = None

        self.tab_grounding_head = nn.Linear(config.d_model * 2, 2)
        self.col_grounding_head = nn.Linear(config.d_model * 2, 2)
        self.val_grounding_head = nn.Linear(config.d_model, 2)

        self.init_metric("loss", "acc", "cls_acc", "cls_loss", "tab_ground_acc", "col_ground_acc", "val_ground_acc",
                         "tab_ground_loss", "col_ground_loss", "val_ground_loss")

        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer, use_fast=False)

        self.generating = False

    def generate(self, *model_args, **model_kwargs):
        self.generating = True
        model_kwargs.pop("tab_index", None)
        model_kwargs.pop("tab_mask", None)
        model_kwargs.pop("col_index", None)
        model_kwargs.pop("col_mask", None)
        model_kwargs.pop("tab_labels", None)
        model_kwargs.pop("col_labels", None)
        model_kwargs.pop("val_labels", None)
        model_kwargs.pop("cls_labels", None)
        res = super().generate(*model_args, **model_kwargs)
        self.generating = False
        return res

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            tab_index: Optional[torch.LongTensor] = None,
            tab_mask: Optional[torch.LongTensor] = None,
            col_index: Optional[torch.LongTensor] = None,
            col_mask: Optional[torch.LongTensor] = None,
            tab_labels: Optional[torch.LongTensor] = None,
            col_labels: Optional[torch.LongTensor] = None,
            val_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cls_labels: Optional[torch.LongTensor] = None,
            meta_data: Dict[str, Any] = None,
            disable_decoder: bool = False,
            **kwargs
    ) -> Union[Dict[str, Tensor], T5Stack]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> import torch

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", return_tensors="pt")).unsqueeze(0)  # Batch size 1
        >>> outputs = model.generate(input_ids)

        >>> print("Generated: {}".format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        Generated: Hello, my dog is cute
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if not self.generating:
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                if self.classification_head is not None:
                    self.classification_head = self.classification_head.to(self.encoder.first_device)
                self.tab_grounding_head = self.tab_grounding_head.to(self.encoder.first_device)
                self.col_grounding_head = self.col_grounding_head.to(self.encoder.first_device)
                self.val_grounding_head = self.val_grounding_head.to(self.encoder.first_device)
                hidden_states = hidden_states.to(self.tab_grounding_head.weight.device)

            if self.num_labels > 0:
                eos_mask = input_ids.eq(self.config.eos_token_id)
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

                # if self.model_parallel:
                #     sentence_representation = sentence_representation.to(self.classification_head.dense.weight.device)

                logits = self.classification_head(sentence_representation)
            else:
                logits = None

            tab_s = torch.gather(hidden_states, dim=1, index=tab_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            tab_e = torch.gather(hidden_states, dim=1, index=tab_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))

            col_s = torch.gather(hidden_states, dim=1, index=col_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            col_e = torch.gather(hidden_states, dim=1, index=col_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))

            tab_logits = self.tab_grounding_head(torch.cat([tab_s, tab_e], dim=-1))
            col_logits = self.col_grounding_head(torch.cat([col_s, col_e], dim=-1))
            val_logits = self.val_grounding_head(hidden_states)
        else:
            logits = None
            tab_logits = None
            col_logits = None
            val_logits = None

        if disable_decoder:
            return Seq2SeqLMPredictionOutput(
                cls_logits=logits if logits is not None else None,
            )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        eval_gen_sentences = None
        if labels is not None:
            label_padding_mask = labels == self.config.pad_token_id
            labels[label_padding_mask] = -1
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom):
            #  Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if not self.training:
                # Generate sentences for BLEU evaluation
                max_output_length = labels.size(1)
                # Greedy decoding.
                eval_gen_sentences = self.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_output_length,
                                                   num_beams=1, do_sample=False)
                eval_gen_sentences = self.tokenizer.batch_decode(eval_gen_sentences, skip_special_tokens=True)

                acc, true_label_num = layers.get_accuracy(lm_logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if cls_labels is not None and self.num_labels > 0:
            cls_loss = self.cls_loss_fct(logits, cls_labels)

            loss = loss + cls_loss if loss is not None else cls_loss

            if not self.training:
                cls_acc, cls_true_label_num = layers.get_accuracy(logits, cls_labels)
                self.eval_metrics.update("cls_acc", cls_acc, n=cls_true_label_num)
                self.eval_metrics.update("cls_loss", cls_loss.item(), n=cls_true_label_num)

        if tab_labels is not None:
            tab_loss = self.cls_loss_fct(tab_logits.view(-1, 2), tab_labels.view(-1))
            col_loss = self.cls_loss_fct(col_logits.view(-1, 2), col_labels.view(-1))
            val_loss = self.cls_loss_fct(val_logits.view(-1, 2), val_labels.view(-1))

            loss = loss + tab_loss + col_loss + val_loss

            if not self.training:
                tab_acc, tab_true_label_num = layers.get_accuracy(tab_logits, tab_labels)
                col_acc, col_true_label_num = layers.get_accuracy(col_logits, col_labels)
                val_acc, val_true_label_num = layers.get_accuracy(val_logits, val_labels)
                self.eval_metrics.update("tab_ground_acc", tab_acc, n=tab_true_label_num)
                self.eval_metrics.update("tab_ground_loss", tab_loss.item(), n=tab_true_label_num)
                self.eval_metrics.update("col_ground_acc", col_acc, n=col_true_label_num)
                self.eval_metrics.update("col_ground_loss", col_loss.item(), n=col_true_label_num)
                self.eval_metrics.update("val_ground_acc", val_acc, n=val_true_label_num)
                self.eval_metrics.update("val_ground_loss", val_loss.item(), n=val_true_label_num)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMPredictionOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            generated_seq=eval_gen_sentences,
            cls_logits=logits if logits is not None else None,
        )


@dataclass
class Seq2SeqLMPredictionOutputHelper1(Seq2SeqLMOutput):
    generated_seq: List[str] = None
    cls_logits: torch.FloatTensor = None


class MT5ForSeq2SeqAndSeqClassificationGroundingV2(MT5ForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: T5Config, tokenizer: str, gradient_checkpointing: bool = False, mat_loss_alpha: float = 1.0,
                 disable_grounding: bool = False):
        config.gradient_checkpointing = gradient_checkpointing
        super().__init__(config)
        self.config = config
        # self.config.gradient_checkpointing = gradient_checkpointing
        self.num_labels = getattr(config, "num_labels", 0)
        if self.num_labels > 0:
            self.classification_head = BartClassificationHead(
                input_dim=config.d_model,
                inner_dim=config.d_ff,
                num_classes=self.num_labels,
                pooler_dropout=config.dropout_rate,
            )
            self.encoder._init_weights(self.classification_head.dense)
            self.encoder._init_weights(self.classification_head.out_proj)
        else:
            self.classification_head = None

        self.disable_grounding = disable_grounding

        self.cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.tab_grounding_head = nn.Linear(config.d_model * 2, 2)
        self.col_grounding_head = nn.Linear(config.d_model * 2, 2)
        self.val_grounding_head = nn.Linear(config.d_model, 2)

        self.mat_loss_alpha = mat_loss_alpha
        self.val_grounding_span = nn.Linear(config.d_model * 2, config.d_model)
        self.col_grounding_span = nn.Linear(config.d_model * 2, config.d_model)

        self.init_metric("loss", "acc", "cls_acc", "cls_loss", "tab_ground_acc", "col_ground_acc", "val_ground_acc",
                         "tab_ground_loss", "col_ground_loss", "val_ground_loss", "val_col_match_loss", "val_col_match_acc")

        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer, use_fast=False)

        self.generating = False

    def generate(self, *model_args, **model_kwargs):
        self.generating = True
        model_kwargs.pop("tab_index", None)
        model_kwargs.pop("tab_mask", None)
        model_kwargs.pop("col_index", None)
        model_kwargs.pop("col_mask", None)
        model_kwargs.pop("val_index", None)
        model_kwargs.pop("val_mask", None)
        model_kwargs.pop("tab_labels", None)
        model_kwargs.pop("col_labels", None)
        model_kwargs.pop("val_labels", None)
        model_kwargs.pop("val_col_match_labels", None)
        model_kwargs.pop("cls_labels", None)
        res = super().generate(*model_args, **model_kwargs)
        self.generating = False
        return res

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            tab_index: Optional[torch.LongTensor] = None,
            tab_mask: Optional[torch.LongTensor] = None,
            col_index: Optional[torch.LongTensor] = None,
            col_mask: Optional[torch.LongTensor] = None,
            val_index: Optional[torch.LongTensor] = None,
            val_mask: Optional[torch.LongTensor] = None,
            tab_labels: Optional[torch.LongTensor] = None,
            col_labels: Optional[torch.LongTensor] = None,
            val_labels: Optional[torch.LongTensor] = None,
            val_col_match_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cls_labels: Optional[torch.LongTensor] = None,
            meta_data: Dict[str, Any] = None,
            disable_decoder: bool = False,
            **kwargs
    ) -> Union[Dict[str, Tensor], T5Stack]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> import torch

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", return_tensors="pt")).unsqueeze(0)  # Batch size 1
        >>> outputs = model.generate(input_ids)

        >>> print("Generated: {}".format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        Generated: Hello, my dog is cute
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if not self.generating:
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                if self.classification_head is not None:
                    self.classification_head = self.classification_head.to(self.encoder.first_device)
                self.tab_grounding_head = self.tab_grounding_head.to(self.encoder.first_device)
                self.col_grounding_head = self.col_grounding_head.to(self.encoder.first_device)
                self.val_grounding_head = self.val_grounding_head.to(self.encoder.first_device)
                self.col_grounding_span = self.col_grounding_span.to(self.encoder.first_device)
                self.val_grounding_span = self.val_grounding_span.to(self.encoder.first_device)
                hidden_states = hidden_states.to(self.tab_grounding_head.weight.device)

            if self.num_labels > 0:
                eos_mask = input_ids.eq(self.config.eos_token_id)
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

                # if self.model_parallel:
                #     sentence_representation = sentence_representation.to(self.classification_head.dense.weight.device)

                logits = self.classification_head(sentence_representation)
            else:
                logits = None

            tab_s = torch.gather(hidden_states, dim=1, index=tab_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            tab_e = torch.gather(hidden_states, dim=1, index=tab_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))

            col_s = torch.gather(hidden_states, dim=1, index=col_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            col_e = torch.gather(hidden_states, dim=1, index=col_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            col_span = torch.cat([col_s, col_e], dim=-1)

            val_s = torch.gather(hidden_states, dim=1, index=val_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            val_e = torch.gather(hidden_states, dim=1, index=val_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            val_span = torch.cat([val_s, val_e], dim=-1)

            tab_logits = self.tab_grounding_head(torch.cat([tab_s, tab_e], dim=-1))
            col_logits = self.col_grounding_head(col_span)
            val_logits = self.val_grounding_head(hidden_states)

            col_span_h = self.col_grounding_span(col_span)
            val_span_h = self.val_grounding_span(val_span)
            mat_scores = torch.einsum("bih,bjh->bij", val_span_h, col_span_h)
            mat_scores = mat_scores + (1 - col_mask[:, None, :].to(mat_scores.dtype)) * -10000.0

        else:
            logits = None
            tab_logits = None
            col_logits = None
            val_logits = None
            mat_scores = None

        if disable_decoder:
            return Seq2SeqLMPredictionOutput(
                cls_logits=logits if logits is not None else None,
            )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        eval_gen_sentences = None
        if labels is not None:
            label_padding_mask = labels == self.config.pad_token_id
            labels[label_padding_mask] = -1
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom):
            #  Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if not self.training:
                # Generate sentences for BLEU evaluation
                max_output_length = labels.size(1)
                # Greedy decoding.
                eval_gen_sentences = self.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_output_length,
                                                   num_beams=1, do_sample=False)
                eval_gen_sentences = self.tokenizer.batch_decode(eval_gen_sentences, skip_special_tokens=True)

                acc, true_label_num = layers.get_accuracy(lm_logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if cls_labels is not None and self.num_labels > 0:
            cls_loss = self.cls_loss_fct(logits, cls_labels)

            loss = loss + cls_loss if loss is not None else cls_loss

            if not self.training:
                cls_acc, cls_true_label_num = layers.get_accuracy(logits, cls_labels)
                self.eval_metrics.update("cls_acc", cls_acc, n=cls_true_label_num)
                self.eval_metrics.update("cls_loss", cls_loss.item(), n=cls_true_label_num)

        if tab_labels is not None and not self.disable_grounding:
            tab_loss = self.cls_loss_fct(tab_logits.view(-1, 2), tab_labels.view(-1))
            col_loss = self.cls_loss_fct(col_logits.view(-1, 2), col_labels.view(-1))
            val_loss = self.cls_loss_fct(val_logits.view(-1, 2), val_labels.view(-1))

            loss = loss + tab_loss + col_loss + val_loss

            if not self.training:
                tab_acc, tab_true_label_num = layers.get_accuracy(tab_logits, tab_labels)
                col_acc, col_true_label_num = layers.get_accuracy(col_logits, col_labels)
                val_acc, val_true_label_num = layers.get_accuracy(val_logits, val_labels)
                self.eval_metrics.update("tab_ground_acc", tab_acc, n=tab_true_label_num)
                self.eval_metrics.update("tab_ground_loss", tab_loss.item(), n=tab_true_label_num)
                self.eval_metrics.update("col_ground_acc", col_acc, n=col_true_label_num)
                self.eval_metrics.update("col_ground_loss", col_loss.item(), n=col_true_label_num)
                self.eval_metrics.update("val_ground_acc", val_acc, n=val_true_label_num)
                self.eval_metrics.update("val_ground_loss", val_loss.item(), n=val_true_label_num)

        if val_col_match_labels is not None and not self.disable_grounding:
            # mat_loss = self.cls_loss_fct(mat_scores.view(-1, mat_scores.size(-1)), val_col_match_labels.view(-1))
            mat_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')(
                mat_scores.view(-1, mat_scores.size(-1)), val_col_match_labels.view(-1))

            mat_true_label_num = (~(val_col_match_labels == -1)).sum().item()

            if mat_true_label_num == 0:
                mat_loss = 0.
            else:
                mat_loss = mat_loss / mat_true_label_num

            loss = loss + self.mat_loss_alpha * mat_loss

            if not self.training:
                mat_acc, mat_true_label_num = layers.get_accuracy(mat_scores, val_col_match_labels)
                self.eval_metrics.update("val_col_match_acc", mat_acc, n=mat_true_label_num)
                self.eval_metrics.update("val_col_match_loss", mat_loss, n=mat_true_label_num)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMPredictionOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            generated_seq=eval_gen_sentences,
            cls_logits=logits if logits is not None else None,
        )


class T5StackEmb(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.name_embed = nn.Embedding(2, config.d_model)
        self.enum_embed = nn.Embedding(2, config.d_model)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            enum_ids=None,
            name_ids=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
            self.enum_embed = self.enum_embed.to(self.first_device)
            self.name_embed = self.name_embed.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.embed_tokens(input_ids) + self.enum_embed(enum_ids) + self.name_embed(name_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class MT5ForSeq2SeqAndSeqClassificationGroundingV2WInputEmb(MT5ForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: T5Config, tokenizer: str, gradient_checkpointing: bool = False, mat_loss_alpha: float = 1.0,
                 disable_grounding: bool = False):
        config.gradient_checkpointing = gradient_checkpointing
        super().__init__(config)
        self.config = config
        # self.config.gradient_checkpointing = gradient_checkpointing

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackEmb(encoder_config, self.shared)

        self.num_labels = getattr(config, "num_labels", 0)
        if self.num_labels > 0:
            self.classification_head = BartClassificationHead(
                input_dim=config.d_model,
                inner_dim=config.d_ff,
                num_classes=self.num_labels,
                pooler_dropout=config.dropout_rate,
            )
            self.encoder._init_weights(self.classification_head.dense)
            self.encoder._init_weights(self.classification_head.out_proj)
        else:
            self.classification_head = None

        self.disable_grounding = disable_grounding

        self.cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.tab_grounding_head = nn.Linear(config.d_model * 2, 2)
        self.col_grounding_head = nn.Linear(config.d_model * 2, 2)
        self.val_grounding_head = nn.Linear(config.d_model, 2)

        self.mat_loss_alpha = mat_loss_alpha
        self.val_grounding_span = nn.Linear(config.d_model * 2, config.d_model)
        self.col_grounding_span = nn.Linear(config.d_model * 2, config.d_model)

        self.init_metric("loss", "acc", "cls_acc", "cls_loss", "tab_ground_acc", "col_ground_acc", "val_ground_acc",
                         "tab_ground_loss", "col_ground_loss", "val_ground_loss", "val_col_match_loss", "val_col_match_acc")

        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer, use_fast=False)

        self.generating = False

    def generate(self, *model_args, **model_kwargs):
        self.generating = True
        model_kwargs.pop("tab_index", None)
        model_kwargs.pop("tab_mask", None)
        model_kwargs.pop("col_index", None)
        model_kwargs.pop("col_mask", None)
        model_kwargs.pop("val_index", None)
        model_kwargs.pop("val_mask", None)
        model_kwargs.pop("tab_labels", None)
        model_kwargs.pop("col_labels", None)
        model_kwargs.pop("val_labels", None)
        model_kwargs.pop("val_col_match_labels", None)
        model_kwargs.pop("cls_labels", None)
        res = super().generate(*model_args, **model_kwargs)
        self.generating = False
        return res

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            enum_ids: Optional[torch.LongTensor] = None,
            name_ids: Optional[torch.LongTensor] = None,
            tab_index: Optional[torch.LongTensor] = None,
            tab_mask: Optional[torch.LongTensor] = None,
            col_index: Optional[torch.LongTensor] = None,
            col_mask: Optional[torch.LongTensor] = None,
            val_index: Optional[torch.LongTensor] = None,
            val_mask: Optional[torch.LongTensor] = None,
            tab_labels: Optional[torch.LongTensor] = None,
            col_labels: Optional[torch.LongTensor] = None,
            val_labels: Optional[torch.LongTensor] = None,
            val_col_match_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cls_labels: Optional[torch.LongTensor] = None,
            meta_data: Dict[str, Any] = None,
            disable_decoder: bool = False,
            **kwargs
    ) -> Union[Dict[str, Tensor], T5Stack]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> import torch

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", return_tensors="pt")).unsqueeze(0)  # Batch size 1
        >>> outputs = model.generate(input_ids)

        >>> print("Generated: {}".format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        Generated: Hello, my dog is cute
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                enum_ids=enum_ids,
                name_ids=name_ids,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if not self.generating:
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                if self.classification_head is not None:
                    self.classification_head = self.classification_head.to(self.encoder.first_device)
                self.tab_grounding_head = self.tab_grounding_head.to(self.encoder.first_device)
                self.col_grounding_head = self.col_grounding_head.to(self.encoder.first_device)
                self.val_grounding_head = self.val_grounding_head.to(self.encoder.first_device)
                self.col_grounding_span = self.col_grounding_span.to(self.encoder.first_device)
                self.val_grounding_span = self.val_grounding_span.to(self.encoder.first_device)
                hidden_states = hidden_states.to(self.tab_grounding_head.weight.device)

            if self.num_labels > 0:
                eos_mask = input_ids.eq(self.config.eos_token_id)
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

                # if self.model_parallel:
                #     sentence_representation = sentence_representation.to(self.classification_head.dense.weight.device)

                logits = self.classification_head(sentence_representation)
            else:
                logits = None

            tab_s = torch.gather(hidden_states, dim=1, index=tab_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            tab_e = torch.gather(hidden_states, dim=1, index=tab_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))

            col_s = torch.gather(hidden_states, dim=1, index=col_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            col_e = torch.gather(hidden_states, dim=1, index=col_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            col_span = torch.cat([col_s, col_e], dim=-1)

            val_s = torch.gather(hidden_states, dim=1, index=val_index[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            val_e = torch.gather(hidden_states, dim=1, index=val_index[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            val_span = torch.cat([val_s, val_e], dim=-1)

            tab_logits = self.tab_grounding_head(torch.cat([tab_s, tab_e], dim=-1))
            col_logits = self.col_grounding_head(col_span)
            val_logits = self.val_grounding_head(hidden_states)

            col_span_h = self.col_grounding_span(col_span)
            val_span_h = self.val_grounding_span(val_span)
            mat_scores = torch.einsum("bih,bjh->bij", val_span_h, col_span_h)
            mat_scores = mat_scores + (1 - col_mask[:, None, :].to(mat_scores.dtype)) * -10000.0

        else:
            logits = None
            tab_logits = None
            col_logits = None
            val_logits = None
            mat_scores = None

        if disable_decoder:
            return Seq2SeqLMPredictionOutput(
                cls_logits=logits if logits is not None else None,
            )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        eval_gen_sentences = None
        if labels is not None:
            label_padding_mask = labels == self.config.pad_token_id
            labels[label_padding_mask] = -1
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom):
            #  Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if not self.training:
                # Generate sentences for BLEU evaluation
                max_output_length = labels.size(1)
                # Greedy decoding.
                eval_gen_sentences = self.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                   enum_ids=enum_ids, name_ids=name_ids,
                                                   max_length=max_output_length,
                                                   num_beams=1, do_sample=False)
                eval_gen_sentences = self.tokenizer.batch_decode(eval_gen_sentences, skip_special_tokens=True)

                acc, true_label_num = layers.get_accuracy(lm_logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if cls_labels is not None and self.num_labels > 0:
            cls_loss = self.cls_loss_fct(logits, cls_labels)

            loss = loss + cls_loss if loss is not None else cls_loss

            if not self.training:
                cls_acc, cls_true_label_num = layers.get_accuracy(logits, cls_labels)
                self.eval_metrics.update("cls_acc", cls_acc, n=cls_true_label_num)
                self.eval_metrics.update("cls_loss", cls_loss.item(), n=cls_true_label_num)

        if tab_labels is not None and not self.disable_grounding:
            tab_loss = self.cls_loss_fct(tab_logits.view(-1, 2), tab_labels.view(-1))
            col_loss = self.cls_loss_fct(col_logits.view(-1, 2), col_labels.view(-1))
            val_loss = self.cls_loss_fct(val_logits.view(-1, 2), val_labels.view(-1))

            loss = loss + tab_loss + col_loss + val_loss

            if not self.training:
                tab_acc, tab_true_label_num = layers.get_accuracy(tab_logits, tab_labels)
                col_acc, col_true_label_num = layers.get_accuracy(col_logits, col_labels)
                val_acc, val_true_label_num = layers.get_accuracy(val_logits, val_labels)
                self.eval_metrics.update("tab_ground_acc", tab_acc, n=tab_true_label_num)
                self.eval_metrics.update("tab_ground_loss", tab_loss.item(), n=tab_true_label_num)
                self.eval_metrics.update("col_ground_acc", col_acc, n=col_true_label_num)
                self.eval_metrics.update("col_ground_loss", col_loss.item(), n=col_true_label_num)
                self.eval_metrics.update("val_ground_acc", val_acc, n=val_true_label_num)
                self.eval_metrics.update("val_ground_loss", val_loss.item(), n=val_true_label_num)

        if val_col_match_labels is not None and not self.disable_grounding:
            # mat_loss = self.cls_loss_fct(mat_scores.view(-1, mat_scores.size(-1)), val_col_match_labels.view(-1))
            mat_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')(
                mat_scores.view(-1, mat_scores.size(-1)), val_col_match_labels.view(-1))

            mat_true_label_num = (~(val_col_match_labels == -1)).sum().item()

            if mat_true_label_num == 0:
                mat_loss = 0.
            else:
                mat_loss = mat_loss / mat_true_label_num

            loss = loss + self.mat_loss_alpha * mat_loss

            if not self.training:
                mat_acc, mat_true_label_num = layers.get_accuracy(mat_scores, val_col_match_labels)
                self.eval_metrics.update("val_col_match_acc", mat_acc, n=mat_true_label_num)
                self.eval_metrics.update("val_col_match_loss", mat_loss, n=mat_true_label_num)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMPredictionOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            generated_seq=eval_gen_sentences,
            cls_logits=logits if logits is not None else None,
        )


def get_device_map(n_gpu: int = 4):
    if n_gpu == 4:
        return {0: [0, 1, 2],
                1: [3, 4, 5, 6, 7, 8, 9],
                2: [10, 11, 12, 13, 14, 15, 16],
                3: [17, 18, 19, 20, 21, 22, 23]}
    elif n_gpu == 8:
        return {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [9, 10, 11],
            4: [12, 13, 14],
            5: [15, 16, 17],
            6: [18, 19, 20],
            7: [21, 22, 23]
        }
    else:
        raise NotImplementedError()
