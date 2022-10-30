from abc import ABC
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from transformers import PretrainedConfig
from transformers.models.dpr.modeling_dpr import DPRPretrainedContextEncoder, DPRPretrainedQuestionEncoder, \
    DPRContextEncoderOutput, DPRQuestionEncoderOutput, DPRConfig, BaseModelOutputWithPooling, DPRPreTrainedModel, BertModel, PreTrainedModel

from general_util.training_utils import batch_to_device


class DPREncoder(DPRPreTrainedModel, ABC):
    base_model_prefix = "bert"

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        assert self.bert.config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert.config.hidden_size, config.projection_dim)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert.config.hidden_size


class DPRContextEncoder(DPRPretrainedContextEncoder, ABC):
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[DPRContextEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples:

        ```python
        >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return DPRContextEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class DPRQuestionEncoder(DPRPretrainedQuestionEncoder, ABC):
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = DPREncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[DPRQuestionEncoderOutput, Tuple[Tensor, ...]]:
        r"""
        Return:

        Examples:

        ```python
        >>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs[1:]
        return DPRQuestionEncoderOutput(
            pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class DensePassageRetrieverConfig(PretrainedConfig):
    def __init__(self, sub_batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.sub_batch_size = sub_batch_size


class PreTrainedDensePassageRetriever(PreTrainedModel, ABC):
    config_class = DensePassageRetrieverConfig
    load_tf_weights = None
    base_model_prefix = ""


class DensePassageRetriever(PreTrainedDensePassageRetriever, ABC):
    def __init__(self, config: DensePassageRetrieverConfig, model_name_or_path, candidate_inputs):
        super().__init__(config)
        self.config = config

        self.ctx_encoder = DPRContextEncoder.from_pretrained(model_name_or_path)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(model_name_or_path)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.candidate_inputs = candidate_inputs
        self.candidate_pooler_outputs = None
        self.sub_batch_size = self.config.sub_batch_size

    def refresh_candidate_representations(self, move_to_cpu: bool = True, cache: bool = True):
        candidate_num = self.candidate_inputs["input_ids"].size(0)

        idx = 0
        pooler_outputs = []
        while idx < candidate_num:
            sub_batch = {k: self.candidate_inputs[k][idx: idx + self.sub_batch_size] for k in self.candidate_inputs.keys()}
            idx += self.sub_batch_size

            sub_batch_output = self.ctx_encoder(**batch_to_device(sub_batch, self.device)).pooler_output

            if move_to_cpu:
                pooler_outputs.append(sub_batch_output.cpu())
            else:
                pooler_outputs.append(sub_batch_output)

        pooler_outputs = torch.cat(pooler_outputs, dim=0)
        if cache:
            self.candidate_pooler_outputs = pooler_outputs
            return
        else:
            return pooler_outputs

    def forward(self, answer_id: Optional[Tensor] = None, **kwargs):
        if self.training:
            if self.candidate_pooler_outputs is not None:
                self.candidate_pooler_outputs = None
        else:
            return self.evaluate(**kwargs)

        if "ctx_input_ids" in kwargs and "que_input_ids" in kwargs:
            ctx_inputs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ctx_")}
            question_inputs = {k[4:]: v for k, v in kwargs.items() if k.startswith("que_")}

            ctx_outputs = self.ctx_encoder(**ctx_inputs)
            question_outputs = self.question_encoder(**question_inputs)

            scores = torch.einsum("ah,bh->ab", question_outputs.pooler_output, ctx_outputs.pooler_output)
            batch_size = scores.size(0)
            labels = torch.arange(batch_size, device=scores.device)

            if answer_id is not None:
                score_mask = answer_id[:, None] == answer_id[None, :]
                # set the diagonal items of `score_mask` to False
                score_mask[labels, labels] = 0
                score_mask = (1 - score_mask.to(dtype=scores.dtype)) * -10000.0
                scores = scores + score_mask

            loss = self.loss_fn(scores, labels)

            return {
                "loss": loss,
                "logits": scores,
            }
        else:
            answer_mask = kwargs.pop("answer_mask", None)
            question_outputs = self.question_encoder(**kwargs).pooler_output
            ctx_outputs = self.refresh_candidate_representations(move_to_cpu=False, cache=False)

            scores = torch.einsum("ah,bh->ab", question_outputs, ctx_outputs)

            if answer_mask is not None:
                scores = scores + answer_mask.to(scores.dtype) * -10000.0

            loss = 0.
            if answer_id is not None:
                loss = self.loss_fn(scores, answer_id)

            return {
                "loss": loss,
                "logits": scores,
            }

    def evaluate(self, **kwargs):
        if self.candidate_pooler_outputs is None:
            self.refresh_candidate_representations()

        question_outputs = self.question_encoder(**kwargs).pooler_output

        candidate_num = self.candidate_pooler_outputs.size(0)

        idx = 0
        scores = []
        while idx < candidate_num:
            sub_candidate_pooler_outputs = self.candidate_pooler_outputs[idx: idx + self.sub_batch_size].to(self.device)
            sub_scores = torch.einsum("ah,bh->ab", question_outputs, sub_candidate_pooler_outputs).cpu()
            scores.append(sub_scores)
            idx += self.sub_batch_size

        scores = torch.cat(scores, dim=1)

        return {
            "loss": 0.,
            "logits": scores[:, :5],
            "scores": scores,
        }
