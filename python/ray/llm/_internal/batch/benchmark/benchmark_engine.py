import ray
import numpy as np

from vllm import LLM
import vllm

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

        input_ids = batch['input_ids']
        ids_np = np.stack(input_ids, axis=0)
        
        prompts = [
            vllm.inputs.data.TokensPrompt(
                prompt_token_ids=token_id,
            ) for token_id in ids_np.tolist()
        ]
        
        result = self.llm.encode(
            prompts=prompts,
            pooling_task="classify",
            pooling_params=vllm.PoolingParams(
                truncate_prompt_tokens=-1,
                task="classify",
            ),
            truncate_prompt_tokens=-1,
        )
        
        output = {
            self.output_column: result,
        }
        
        return output

def main():
    ray.init()
    model_name = "HuggingFaceTB/fineweb-edu-classifier"
    dataset_size = 100_000  # 4GB for 1_000_000 items

    ds = ray.data.from_items([{"prompt": "Hello", "input_ids": [1] * 512} for _ in range(dataset_size)])
    ds = ds.repartition(8)

    classified_ds = ds.map_batches(
        vLLMSyncWrapper(model_name),
        batch_size=512,
        num_gpus=1,
        concurrency=1,
    )

    classified_ds.materialize()

if __name__ == "__main__":
    main()