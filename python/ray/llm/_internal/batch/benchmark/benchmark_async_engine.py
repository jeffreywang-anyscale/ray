import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    TokenizerStageConfig,
)
from dataset import ShareGPTDataset

ray.init()

model_name = "HuggingFaceTB/fineweb-edu-classifier"
dataset_size = 1_000_000

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
    tokenize_stage=TokenizerStageConfig(enabled=True),
    detokenize_stage=DetokenizeStageConfig(enabled=False),
)

processor = build_processor(
    processor_config,
    preprocess=lambda row: dict(
        prompt=row['prompt'],
        # tokenized_prompt=row['input_ids'],
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
ds = processor(ds)
ds = ds.materialize()
