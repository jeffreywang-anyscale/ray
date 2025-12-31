import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    TokenizerStageConfig,
)

ray.init()

# model_name = "facebook/opt-1.3b"
model_name = "HuggingFaceTB/fineweb-edu-classifier"
dataset_size = 100_000

processor_config = vLLMEngineProcessorConfig(
    model_source=model_name,
    engine_kwargs=dict(
        enforce_eager=True,
        max_model_len=512,
    ),
    task_type="classify",  # Set at config level, not in engine_kwargs
    batch_size=512,
    concurrency=1,
    chat_template_stage=ChatTemplateStageConfig(enabled=False),
    tokenize_stage=TokenizerStageConfig(enabled=False),
    detokenize_stage=DetokenizeStageConfig(enabled=False),
)

processor = build_processor(
    processor_config,
    preprocess=lambda row: dict(
        prompt=row['prompt'],
        tokenized_prompt=row['input_ids'],
        pooling_params={
            "truncate_prompt_tokens": -1,
        }
    ),
    postprocess=lambda row: {
        "probs": float(row['embeddings'][0])
        if row.get('embeddings') is not None and len(row['embeddings']) > 0
        else None,
    },
)

ds = ray.data.from_items([{"prompt": "Hello", "input_ids": [1] * 512} for _ in range(dataset_size)])
ds = ds.repartition(8)
ds = processor(ds)
ds = ds.materialize()
# print(ds.take(1))
