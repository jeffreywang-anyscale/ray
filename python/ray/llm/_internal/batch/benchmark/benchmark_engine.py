import argparse

import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    TokenizerStageConfig,
)
from dataset import ShareGPTDataset
from sync_engine_wrapper import Tokenizer, vLLMSyncWrapper

from ray.runtime_env import RuntimeEnv


def main(args):
    # Add Hugging Face token to the runtime environment
    ray.init(runtime_env=RuntimeEnv(env_vars={}))
    model_name = args.model_name
    dataset_size = args.dataset_size

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
        Tokenizer,
        batch_size=args.batch_size,
        zero_copy_batch=True,
        num_cpus=1,
        compute=ray.data.ActorPoolStrategy(size=50),
        batch_format="pandas",
        fn_constructor_kwargs={
            "model_path": model_name,
        },
    )

    if args.sync_engine:
        ds = ds.map_batches(
            vLLMSyncWrapper,
            batch_size=args.batch_size,
            batch_format="pandas",
            zero_copy_batch=True,
            num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=1),
            fn_constructor_kwargs={
                "model_path": model_name,
                "mode": args.mode,
                "output_column": "probs" if args.mode == "classify" else "generated_text",
                "max_decode_tokens": args.max_decode_tokens,
                "ignore_eos": args.ignore_eos,
            },
        )
    else:
        processor_config = vLLMEngineProcessorConfig(
            model_source=model_name,
            engine_kwargs=dict(
                enforce_eager=True,
                max_model_len=512,
            ),
            task_type=args.mode,  # "classify" or "generate"
            batch_size=args.batch_size,
            concurrency=1,
            chat_template_stage=ChatTemplateStageConfig(enabled=False),
            tokenize_stage=TokenizerStageConfig(enabled=False),
            detokenize_stage=DetokenizeStageConfig(enabled=False),
        )

        if args.mode == "classify":
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
        else:  # generate mode
            processor = build_processor(
                processor_config,
                preprocess=lambda row: dict(
                    prompt=row['prompt'],
                    tokenized_prompt=row['input_ids'],
                    sampling_params={
                        "max_tokens": args.max_decode_tokens,
                        "ignore_eos": args.ignore_eos,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    }
                ),
                postprocess=lambda row: {
                    "generated_text": row.get('generated_text', ''),
                },
            )
        ds = processor(ds)

    ds = ds.materialize()
    print(ds.take(1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark async vLLM engine")
    parser.add_argument(
        "--sync-engine",
        action="store_true",
        default=False,
        help="Use synchronous vLLM engine instead of async processor",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/fineweb-edu-classifier",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1_000,
        help="Dataset size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        # default=1024,
        default=131702,
        # default=526808,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classify",
        choices=["classify", "generate"],
        help="Mode to run the benchmark in. classify: classify the prompt, generate: generate the response",
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (only for generate mode)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Ignore EOS token during generation (only for generate mode)",
    )
    args = parser.parse_args()
    main(args)
