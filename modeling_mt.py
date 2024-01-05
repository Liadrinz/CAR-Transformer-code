import torch
import warnings
import copy

from dataclasses import dataclass
from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.utils import logging
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.models.mbart.tokenization_mbart import MBartTokenizer
from transformers.models.mbart.modeling_mbart import (
    MBartForConditionalGeneration,
    MBartModel,
    MBartEncoder,
    MBartDecoder,
    shift_tokens_right,
)
from modeling import CrossAttentionReinforcing
# from transformers.models.mt5.modeling_mt5 import (
#     MT5ForConditionalGeneration,
#     MT5Config,
#     MT5Stack,
#     __HEAD_MASK_WARNING_MSG,
# )


logger = logging.get_logger(__name__)


@dataclass
class ParallelSeq2SeqModelOutput(Seq2SeqModelOutput):
    
    last_hidden_state_para: torch.FloatTensor = None
    past_key_values_para: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_para_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_para_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_para: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ParallelSeq2SeqLMOutput(Seq2SeqLMOutput):
    
    logits_para: torch.FloatTensor = None
    past_key_values_para: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_para_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_para_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_para: Optional[Tuple[torch.FloatTensor]] = None


class MBartParallelModel(MBartModel):
    
    def __init__(self, config: MBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)
        self.decoder_para = MBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()
    
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
        # ==== para_decoder ====
        decoder_para_input_ids: Optional[torch.LongTensor] = None,
        decoder_para_attention_mask: Optional[torch.LongTensor] = None,
        decoder_para_head_mask: Optional[torch.Tensor] = None,
        cross_attn_para_head_mask: Optional[torch.Tensor] = None,
        past_key_values_para: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_para_inputs_embeds: Optional[torch.FloatTensor] = None,
        # ==== para_decoder ====
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[ParallelSeq2SeqModelOutput, Tuple[torch.FloatTensor]]:
        is_para = decoder_para_input_ids is not None or decoder_para_inputs_embeds is not None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if is_para:
            decoder_para_outputs = self.decoder_para(
                input_ids=decoder_para_input_ids,
                attention_mask=decoder_para_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_para_head_mask,
                cross_attn_head_mask=cross_attn_para_head_mask,
                past_key_values=past_key_values_para,
                inputs_embeds=decoder_para_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if not return_dict:
            return decoder_outputs + (decoder_para_outputs if is_para else ()) + encoder_outputs

        return ParallelSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # ==== decoder_para ====
            last_hidden_state_para=decoder_para_outputs.last_hidden_state if is_para else None,
            past_key_values_para=decoder_para_outputs.past_key_values if is_para else None,
            decoder_para_hidden_states=decoder_para_outputs.hidden_states if is_para else None,
            decoder_para_attentions=decoder_para_outputs.attentions if is_para else None,
            cross_attentions_para=decoder_para_outputs.cross_attentions if is_para else None,
            # ==== decoder_para ====
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MBartForParallelGeneration(MBartForConditionalGeneration):
    
    def __init__(self, config: MBartConfig, pg_weight: float = 1.0):
        super().__init__(config)
        self.pg_weight = pg_weight
        self.model = MBartParallelModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
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
        # ==== para_decoder ====
        decoder_para_input_ids: Optional[torch.LongTensor] = None,
        decoder_para_attention_mask: Optional[torch.LongTensor] = None,
        decoder_para_head_mask: Optional[torch.Tensor] = None,
        cross_attn_para_head_mask: Optional[torch.Tensor] = None,
        past_key_values_para: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_para_inputs_embeds: Optional[torch.FloatTensor] = None,
        # ==== para_decoder ====
        labels: Optional[torch.LongTensor] = None,
        labels_para: Optional[torch.LongTensor] = None,
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
        is_para = decoder_para_input_ids is not None or decoder_para_inputs_embeds is not None or labels_para is not None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        
        if labels_para is not None:
            if decoder_para_input_ids is None and decoder_para_inputs_embeds is None:
                decoder_para_input_ids = shift_tokens_right(labels_para, self.config.pad_token_id)

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
            # ==== para_decoder ====
            decoder_para_input_ids=decoder_para_input_ids,
            decoder_para_attention_mask=decoder_para_attention_mask,
            decoder_para_head_mask=decoder_para_head_mask,
            cross_attn_para_head_mask=cross_attn_para_head_mask,
            past_key_values_para=past_key_values_para,
            decoder_para_inputs_embeds=decoder_para_inputs_embeds,
            # ==== para_decoder ====
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        lm_logits_para = None
        if is_para:
            lm_logits_para = self.lm_head(outputs.last_hidden_state_para) + self.final_logits_bias

        total_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss_fct = MSELoss()
        
        if is_para and labels_para is not None:
            loss_fct = CrossEntropyLoss()
            loss_para = loss_fct(lm_logits_para.view(-1, self.config.vocab_size), labels_para.view(-1))
            if total_loss is not None:
                total_loss = total_loss + self.pg_weight * loss_para
            else:
                total_loss = loss_para
        
        if not return_dict:
            output = (lm_logits, lm_logits_para,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ParallelSeq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            logits_para=lm_logits_para,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            # ==== decoder_para ====
            past_key_values_para=outputs.past_key_values_para,
            decoder_para_hidden_states=outputs.decoder_para_hidden_states,
            decoder_para_attentions=outputs.decoder_para_attentions,
            cross_attentions_para=outputs.cross_attentions_para,
            # ==== decoder_para ====
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class MBartForParallelGenerationCAR(MBartForParallelGeneration, CrossAttentionReinforcing):
    
    def __init__(
        self,
        config: MBartConfig,
        tokenizer: MBartTokenizer = None,
        src_lang: str = None,
        pg_weight: float = 1.0,
        self_critic: bool = True,
    ):
        super().__init__(config, pg_weight)
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
        # ==== para_decoder ====
        decoder_para_input_ids: Optional[torch.LongTensor] = None,
        decoder_para_attention_mask: Optional[torch.LongTensor] = None,
        decoder_para_head_mask: Optional[torch.Tensor] = None,
        cross_attn_para_head_mask: Optional[torch.Tensor] = None,
        past_key_values_para: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_para_inputs_embeds: Optional[torch.FloatTensor] = None,
        # ==== para_decoder ====
        labels: Optional[torch.LongTensor] = None,
        labels_para: Optional[torch.LongTensor] = None,
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
        is_para = decoder_para_input_ids is not None or decoder_para_inputs_embeds is not None or labels_para is not None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        
        if labels_para is not None:
            if decoder_para_input_ids is None and decoder_para_inputs_embeds is None:
                decoder_para_input_ids = shift_tokens_right(labels_para, self.config.pad_token_id)

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
            # ==== para_decoder ====
            decoder_para_input_ids=decoder_para_input_ids,
            decoder_para_attention_mask=decoder_para_attention_mask,
            decoder_para_head_mask=decoder_para_head_mask,
            cross_attn_para_head_mask=cross_attn_para_head_mask,
            past_key_values_para=past_key_values_para,
            decoder_para_inputs_embeds=decoder_para_inputs_embeds,
            # ==== para_decoder ====
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        lm_logits_para = None
        if is_para:
            lm_logits_para = self.lm_head(outputs.last_hidden_state_para) + self.final_logits_bias

        total_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss_fct = MSELoss()
        
        if is_para and labels_para is not None:
            loss_fct = CrossEntropyLoss()
            loss_para = loss_fct(lm_logits_para.view(-1, self.config.vocab_size), labels_para.view(-1))
            cross_attentions = outputs.cross_attentions[-1].mean(dim=1)
            policy_loss, rewards, log_probs = self.compute_policy_loss(cross_attentions, input_ids, attention_mask, labels_para)
            if total_loss is not None:
                total_loss = total_loss + self.pg_weight * loss_para + self.pg_weight * policy_loss
                print(f"lm_loss: {total_loss.item():.4f}, policy_loss: {policy_loss.item():.4f}")
                print("rewards:", [round(i, 4) for i in rewards.tolist()])
                print("log_probs:", [round(i, 4) for i in log_probs.tolist()])
            else:
                total_loss = self.pg_weight * loss_para + self.pg_weight * policy_loss
        
        if not return_dict:
            output = (lm_logits, lm_logits_para,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ParallelSeq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            logits_para=lm_logits_para,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            # ==== decoder_para ====
            past_key_values_para=outputs.past_key_values_para,
            decoder_para_hidden_states=outputs.decoder_para_hidden_states,
            decoder_para_attentions=outputs.decoder_para_attentions,
            cross_attentions_para=outputs.cross_attentions_para,
            # ==== decoder_para ====
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
