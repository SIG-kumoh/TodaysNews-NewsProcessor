import logging
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, configuration_utils
configuration_utils.logger.setLevel(logging.ERROR)


class KoBARTSummarizer:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        # TODO USE CONFIG FILE
        self.model = BartForConditionalGeneration.from_pretrained('resources/summary_model')

    def summarize(self, text: str) -> str:
        text = text.replace('\n', ' ')
        raw_input_ids = self.tokenizer.encode(text)
        input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]

        # 최대 길이 넘으면 자름
        if len(input_ids) > 1026:
            input_ids = input_ids[:1026]

        summary_ids = self.model.generate(torch.tensor([input_ids]), num_beams=4, max_length=1024, eos_token_id=1)
        return self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
