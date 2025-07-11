import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import BatchEncoding
import xgrammar as xgr

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    Collator,
    flatten_image_list,
    handle_stop_sequences,
    pad_and_concat,
    replace_placeholders,
    resize_image,
    stop_sequences_criteria,
)

eval_logger = logging.getLogger(__name__)


@register_model("hf-structured")
class HFStructuredLM(HFLM):
    """
    An abstracted Hugging Face model class for structured LMs like Llava and Idefics.
    """
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        grammar_file_path: str,
        **kwargs,
    ):
        print("pretrained:", pretrained)
        print("kwargs:", kwargs)
        super().__init__(pretrained, **kwargs)
        # TODO: default json for now
        self._create_logits_processor()

    def _create_logits_processor(self):
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.config.vocab_size)
        compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
        self.compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()
        self.logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)

    # def _model_call(self, inps, attn_mask=None, labels=None):
    #     """
    #     :param inps: torch.Tensor
    #         A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
    #         [batch, sequence_ctx]. the size of sequence may vary from call to call
    #     :param attn_mask: torch.Tensor, optional
    #         A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
    #         (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
    #     :param labels: torch.Tensor, optional
    #         A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
    #         (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
    #     :return
    #         A torch tensor of shape [batch, sequence, vocab] with the
    #     logits returned from the model's decoder
    #     """
    #     with torch.no_grad():
    #         if attn_mask is not None or labels is not None:
    #             assert attn_mask is not None and labels is not None
    #             assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
    #             return self.model(
    #                 input_ids=inps, attention_mask=attn_mask, labels=labels
    #             ).logits
    #         else:
    #             assert self.AUTO_MODEL_CLASS in (
    #                 transformers.AutoModelForCausalLM,
    #                 transformers.AutoModelForVision2Seq,
    #             )
    #             actual_batch_size = inps.shape[0]
    #             matchers = [
    #                 xgr.GrammarMatcher(self.compiled_grammar)
    #                 for i in range(actual_batch_size)
    #             ]
    #             token_bitmask = xgr.allocate_token_bitmask(actual_batch_size, self.tokenizer.vocab_size)

    #             # This for loop is parallelizable using threading.Thread. But estimate
    #             # the overhead in your engine.
    #             logits = self.model(inps).logits

    #             print(logits.shape)
    #             print(logits)
    #             print(logits.dtype)

    #             logits = logits.to(torch.float32)
    #             last_logits = logits[:, -1, :]
    #             for i in range(actual_batch_size):
    #                 matchers[i].fill_next_token_bitmask(token_bitmask, i)
    #             xgr.apply_token_bitmask_inplace(last_logits, token_bitmask.to(logits.device))

    #             logits[:, -1, :] = last_logits

    #             return logits

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS in (
                    transformers.AutoModelForCausalLM,
                    transformers.AutoModelForVision2Seq,
                )

                logits = self.model(inps).logits

                logits = self.logits_processor(logits)

                return logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            logits_processor=[self.logits_processor]
            **generation_kwargs,
        )
