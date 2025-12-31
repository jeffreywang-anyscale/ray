import ray
import numpy as np

from vllm import LLM
import vllm

from dataset import ShareGPTDataset

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, batch) -> dict:
        prompts = batch['prompt']
        prompts = prompts.tolist()
        
        tokenized = self.tokenizer(prompts)
        return {'input_ids': tokenized['input_ids']}

class vLLMSyncWrapper:
    def __init__(self, model_path: str, output_column: str = "probs"):
        self.model_path = model_path
        self.output_column = output_column
        self.llm = None

    def __call__(self, batch) -> dict:
        if self.llm is None:
            self.llm = LLM(
                model=self.model_path,
                enforce_eager=True,
                max_model_len=512,
            )

        input_ids = batch['input_ids'].tolist()
        
        prompts = [
            vllm.inputs.data.TokensPrompt(
                prompt_token_ids=token_id_list.tolist(),
            ) for token_id_list in input_ids
        ]

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
        
        # result = self.llm.encode(
        #     prompts=prompts,
        #     pooling_task="classify",
        #     pooling_params=vllm.PoolingParams(
        #         truncate_prompt_tokens=-1,
        #         task="classify",
        #     ),
        #     truncate_prompt_tokens=-1,
        # )
        
        # output = {
        #     self.output_column: result,
        # }
        
        return output

def main():
    ray.init()
    model_name = "HuggingFaceTB/fineweb-edu-classifier"
    dataset_size = 1_000_000  # 4GB for 1_000_000 items

    dataset = ShareGPTDataset(
        dataset_path="/tmp/data/Code-feedback-sharegpt-renamed",
        seed=0,
        hf_dataset_id="Crystalcareai/Code-feedback-sharegpt-renamed",
        hf_split="train",
        truncate_prompt=512,
    )
    prompts = dataset.sample(dataset_size)

    ds = ray.data.from_items(prompts)
    # ds = ds.repartition(8)

    ds = ds.map_batches(
        Tokenizer(model_name),
        batch_size=512,
        zero_copy_batch=True,
        fn_constructor_kwargs={
            "model_path": model_name,
        },
    )

    classified_ds = ds.map_batches(
        vLLMSyncWrapper(model_name),
        batch_size=512,
        num_gpus=1,
        concurrency=1,
    )

    classified_ds.materialize()

if __name__ == "__main__":
    main()