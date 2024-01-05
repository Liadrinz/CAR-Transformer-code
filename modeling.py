import torch
import warnings
import json

from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from torch.distributions import Bernoulli
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.models.mbart.tokenization_mbart import MBartTokenizer
from transformers.models.mbart.modeling_mbart import (
    MBartForConditionalGeneration,
    shift_tokens_right,
)
from transformers.models.mt5.configuration_mt5 import MT5Config
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG
from rouge_score.rouge_scorer import RougeScorer


logger = logging.get_logger(__name__)
LANG_MAP = json.load(open("lang_map.json"))
LANG_MAP_R = {v: k for k, v in LANG_MAP.items()}


class CrossAttentionReinforcing:
    
    def __init__(self, config: PretrainedConfig, tokenizer: PreTrainedTokenizer, src_lang: str, pg_weight: float, self_critic: bool) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True, lang=src_lang)
        self.pg_weight = pg_weight
        self.self_critic = self_critic

    def reward_fn(self, select_ids: torch.LongTensor, ml_labels: torch.LongTensor) -> torch.FloatTensor:
        sel_output = self.tokenizer.batch_decode(select_ids, skip_special_tokens=True)
        ml_summary = self.tokenizer.batch_decode(ml_labels, skip_special_tokens=True)
        rouge1 = torch.tensor([self.rouge_scorer.score(ref, hyp)["rouge1"].fmeasure for ref, hyp in zip(ml_summary, sel_output)], device=select_ids.device)
        rouge2 = torch.tensor([self.rouge_scorer.score(ref, hyp)["rouge2"].fmeasure for ref, hyp in zip(ml_summary, sel_output)], device=select_ids.device)
        rougel = torch.tensor([self.rouge_scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(ml_summary, sel_output)], device=select_ids.device)
        rewards = (rouge1 + rouge2 + rougel) * 100 / 3
        return rewards
    
    def get_select_ids_by_bool_selection(self, input_ids: torch.LongTensor, sel: torch.LongTensor):
        select_ids = input_ids[:, 1:].clone()
        select_ids[~sel.bool()] = self.config.pad_token_id
        rank_indices = (select_ids == self.config.pad_token_id).float().argsort(dim=-1)
        select_ids = select_ids.gather(-1, rank_indices)
        min_pad_num = (select_ids == self.config.pad_token_id).sum(dim=-1).min()
        if min_pad_num > 0:
            select_ids = select_ids[:, :-min_pad_num]
        return select_ids
    
    def compute_policy_loss(self, cross_attentions: torch.FloatTensor, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, ml_labels: torch.LongTensor):
        cross_attentions_norm = cross_attentions[:, 1:, 1:] / cross_attentions[:, 1:, 1:].sum(dim=-1, keepdim=True)
        source_awareness = cross_attentions_norm.mean(dim=1)
        sel_dist = Bernoulli(source_awareness)
        sel = sel_dist.sample()
        log_probs = sel_dist.log_prob(sel).sum(dim=-1) / attention_mask.sum(dim=-1)
        select_ids = self.get_select_ids_by_bool_selection(input_ids, sel)
        rewards = self.reward_fn(select_ids, ml_labels).detach()
        if self.self_critic:
            _, greedy_sel = source_awareness.topk(min(ml_labels.size(1), source_awareness.size(1)), dim=-1)
            greedy_select_ids = input_ids[:, 1:].gather(-1, greedy_sel)
            # greedy_sel = (source_awareness > 0.5).long()
            # greedy_select_ids = self.get_select_ids_by_bool_selection(input_ids, greedy_sel)
            baselines = self.reward_fn(greedy_select_ids, ml_labels).detach()
            rewards = rewards - baselines
        policy_loss = (-log_probs * rewards).mean()
        return policy_loss, rewards, log_probs


class MBartForConditionalGenerationCAR(MBartForConditionalGeneration, CrossAttentionReinforcing):
    
    def __init__(
        self,
        config: MBartConfig,
        tokenizer: MBartTokenizer = None,
        src_lang: str = None,
        pg_weight: float = 1.0,
        self_critic: bool = True,
    ):
        super().__init__(config)
        CrossAttentionReinforcing.__init__(self, config, tokenizer, src_lang, pg_weight, self_critic)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ml_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        loss = None
        if ml_labels is not None:
            cross_attentions = outputs.cross_attentions[-1].mean(dim=1)
            policy_loss, rewards, log_probs = self.compute_policy_loss(cross_attentions, input_ids, attention_mask, ml_labels)
            if lm_loss is None:
                loss = self.pg_weight * policy_loss
            else:
                loss = lm_loss + self.pg_weight * policy_loss
                print(f"lm_loss: {lm_loss.item():.4f}, policy_loss: {policy_loss.item():.4f}")
                print("rewards:", [round(i, 4) for i in rewards.tolist()])
                print("log_probs:", [round(i, 4) for i in log_probs.tolist()])
        else:
            loss = lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class MT5ForConditionalGenerationCAR(MT5ForConditionalGeneration, CrossAttentionReinforcing):
    
    def __init__(
        self,
        config: MT5Config,
        tokenizer: T5Tokenizer = None,
        src_lang: str = None,
        pg_weight: float = 1.0,
        self_critic: bool = True,
    ):
        super().__init__(config)
        CrossAttentionReinforcing.__init__(self, config, tokenizer, src_lang, pg_weight, self_critic)
    
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
        ml_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

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
            output_attentions=True,
            output_hidden_states=True,
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
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        
        loss = None
        if ml_labels is not None:
            cross_attentions = decoder_outputs.cross_attentions[-1].mean(dim=1)
            policy_loss, rewards, log_probs = self.compute_policy_loss(cross_attentions, input_ids, attention_mask, ml_labels)
            if lm_loss is None:
                loss = self.pg_weight * policy_loss
            else:
                loss = lm_loss + self.pg_weight * policy_loss
                print(f"lm_loss: {lm_loss.item():.2f}, policy_loss: {policy_loss.item():.2f}")
                print("rewards:", [round(i, 4) for i in rewards.tolist()])
                print("log_probs:", [round(i, 4) for i in log_probs.tolist()])
        else:
            loss = lm_loss
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )