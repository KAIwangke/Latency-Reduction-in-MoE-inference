TOKEN = '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'
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2

pipeline = None

GLOBAL_SERVER = None


class Server:
    def __init__(self, approx_model_name, target_model_name) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logging.info("begin load models")
        self._small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, trust_remote_code=True,
                                                                 token=TOKEN).to(self._device)
        self._large_model = AutoModelForCausalLM.from_pretrained(target_model_name, trust_remote_code=True,
                                                                 token=TOKEN).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        logging.info("fininsh load models")

        self.num_tokens = 40
        self.top_k = 10
        self.top_p = 0.9

    def process_request(self, request: str) -> torch.Tensor:
        input_str = request['prompt']
        logging.info(f"recieve request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        output = speculative_sampling(input_ids,
                                      self._small_model,
                                      self._large_model, self.num_tokens,
                                      top_k=self.top_k,
                                      top_p=self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":
    server = Server(
        approx_model_name="Maykeye/TinyLLama-v0",
        target_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    request_data={ "prompt": "hello my name is"}
    # Perform inference
    result = server.process_request(request_data)
    print(result)
