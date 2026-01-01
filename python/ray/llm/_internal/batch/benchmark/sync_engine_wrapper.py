import pandas as pd

from vllm import LLM
import vllm

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, batch: pd.DataFrame) -> dict:
        prompts = batch['prompt'].tolist()

        tokenized = self.tokenizer(prompts)
        return {'prompt': prompts, 'input_ids': tokenized['input_ids']}

class vLLMSyncWrapper:
    def __init__(
        self,
        model_path: str,
        mode: str = "classify",
        output_column: str = "probs",
        max_decode_tokens: int = 100,
        ignore_eos: bool = False,
    ):
        self.model_path = model_path
        self.mode = mode
        self.output_column = output_column
        self.max_decode_tokens = max_decode_tokens
        self.ignore_eos = ignore_eos
        
        # Initialize LLM with appropriate task
        task = "classify" if mode == "classify" else None
        self.llm = LLM(
            model=self.model_path,
            enforce_eager=True,
            max_model_len=512,
            task=task,
        )

    def __call__(self, batch: pd.DataFrame) -> dict:
        input_ids = batch['input_ids'].tolist()
        
        prompts = [
            vllm.inputs.data.TokensPrompt(
                prompt_token_ids=token_id_list.tolist(),
            ) for token_id_list in input_ids
        ]

        if self.mode == "classify":
            result = self.llm.classify(
                prompts=prompts,
                pooling_params=vllm.PoolingParams(
                    truncate_prompt_tokens=-1,
                    task="classify",
                ),
            )
            output = {
                self.output_column: [out.outputs.probs for out in result]
            }
        elif self.mode == "generate":
            result = self.llm.generate(
                prompts=prompts,
                sampling_params=vllm.SamplingParams(
                    max_tokens=self.max_decode_tokens,
                    ignore_eos=self.ignore_eos,
                    temperature=1.0,
                    top_p=1.0,
                ),
            )
            output = {
                self.output_column: [out.outputs[0].text for out in result]
            }
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        return output
