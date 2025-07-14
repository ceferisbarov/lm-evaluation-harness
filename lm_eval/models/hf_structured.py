import logging
from typing import Union

import transformers
import xgrammar as xgr

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    stop_sequences_criteria,
)


eval_logger = logging.getLogger(__name__)


@register_model("hf-structured")
class HFStructuredLM(HFLM):
    """
    An abstracted Hugging Face model class for structured LMs.
    """

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        grammar_file_path: str,
        **kwargs,
    ):
        super().__init__(pretrained, **kwargs)
        # TODO: default json for now
        self._create_compiled_grammar()

    def _create_compiled_grammar(self):
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.config.vocab_size
        )
        compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
        self.compiled_grammar: xgr.CompiledGrammar = (
            compiler.compile_builtin_json_grammar()
        )

    def _get_logits_processor(self):
        return xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)

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
            logits_processor=[self._get_logits_processor()],
            **generation_kwargs,
        )
