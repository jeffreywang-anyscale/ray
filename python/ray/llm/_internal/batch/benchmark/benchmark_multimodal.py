import subprocess
import sys
import os
import json
import random
from time import perf_counter
from dataclasses import dataclass

# # Dependency setup
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "--upgrade", "transformers", "datasets", "huggingface_hub"]
# )
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ray[llm]"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])

import ray
from ray.data.llm import (
    vLLMEngineProcessorConfig,
    build_processor,
    PrepareMultimodalStageConfig,
    ChatTemplateStageConfig,
    TokenizerStageConfig,
    DetokenizeStageConfig,
)

def video_preprocess(row: dict) -> dict:
    """
    Preprocessing function for video-language model inputs.

    Converts dataset rows from ShareGPTVideo QA format into the format expected by the VLM.
    Only sends the human message from conversations (GPT responses are excluded).
    
    The text field already contains only the human message (extracted during dataset loading).
    """
    # Get the human text prompt (already extracted, no GPT responses)
    text_prompt = row.get("text", "")
    
    # If no text prompt found, use default
    if not text_prompt:
        text_prompt = "What happens in this video?"
    
    content = [
        {
            "type": "text",
            "text": text_prompt,
        },
    ]

    # Add video content only 50% of the time to mix text-only and video+text requests.
    if random.random() < 0 and "video_url" in row:
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": row["video_url"]},
            }
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that analyzes videos. "
                    "Watch the video carefully and answer questions about it."
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        "sampling_params": {
            "temperature": 0.3,
            "max_tokens": 50,  # For QA responses
            # "ignore_eos": True,
            "detokenize": False,
        },
        # Optional: Multimodal processor kwargs for video processing
        "mm_processor_kwargs": dict(
            min_pixels=28 * 28,
            max_pixels=1280 * 28 * 28,
            fps=1,
        ),
    }


def video_postprocess(row: dict) -> dict:
    """Postprocess video QA results."""
    return {
        "id": row.get("id", ""),
        "video": row.get("video", ""),
        "text": row.get("text", ""),  # Preserve the prompt text
        "caption": row.get("generated_text", ""),
        "resp": row.get("generated_text", ""),
    }


@dataclass
class VideoCaptioningBenchmarkResult:
    """Results from video QA throughput benchmark."""
    samples: int
    elapsed_s: float
    batch_size: int
    concurrency: int
    
    @property
    def throughput(self) -> float:
        """Calculate throughput in samples per second."""
        return self.samples / self.elapsed_s if self.elapsed_s > 0 else 0.0
    
    @property
    def throughput_videos_per_sec(self) -> float:
        """Alias for throughput (videos per second)."""
        return self.throughput
    
    def show(self) -> None:
        """Display benchmark results."""
        print("\n" + "=" * 60)
        print("VIDEO QA THROUGHPUT BENCHMARK")
        print("=" * 60)
        print(f"Samples processed: {self.samples}")
        print(f"Batch size       : {self.batch_size}")
        print(f"Concurrency      : {self.concurrency}")
        print(f"Time (s)         : {self.elapsed_s:.2f}")
        print(f"Throughput       : {self.throughput:.2f} videos/s")
        print("=" * 60)




def load_video_dataset(max_samples: int = None):
    """
    Load video QA dataset combining:
    - Videos from ShareGPTVideo/train_raw_video/activitynet/chunk_*.tar.gz
    - Prompts from ShareGPTVideo/train_video_and_instruction/video_instruction/train/qa/chatgpt_qa_900k.jsonl
    
    Strategy (download-first):
    1) Download/extract videos (all activitynet chunks, stop early once we reach max_samples).
    2) Build a prompt lookup from JSONL (video_id -> list of prompts).
    3) For each video file:
         - If prompts exist, create one row per prompt.
         - Otherwise, create a default prompt: "Describe this video in detail."
    """
    try:
        from huggingface_hub import hf_hub_download
        import tarfile
        from pathlib import Path
        import ray.data

        dataset_name = "ShareGPTVideo/train_video_and_instruction"
        jsonl_path = "video_instruction/train/qa/chatgpt_qa_900k.jsonl"
        raw_video_dataset_name = "ShareGPTVideo/train_raw_video"
        extract_dir = "/tmp/sharegpt_videos"
        os.makedirs(extract_dir, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Step 1: Download/extract videos first
        # ------------------------------------------------------------------ #
        video_files_map = {}  # video_id (stem) -> path

        # helper to record existing mp4s
        def index_existing_mp4s():
            for video_file in Path(extract_dir).rglob("*.mp4"):
                vid = video_file.stem
                video_files_map[vid] = str(video_file)

        index_existing_mp4s()
        print(f"Found {len(video_files_map)} existing video files in {extract_dir}")

        target_samples = max_samples or 0
        need_more = target_samples == 0 or len(video_files_map) < target_samples

        if need_more:
            tarballs_to_try = [f"activitynet/chunk_{i}.tar.gz" for i in range(24)]
            for tarball_path in tarballs_to_try:
                if target_samples and len(video_files_map) >= target_samples:
                    break
                print(f"\nTrying tarball: {tarball_path}")
                try:
                    tar_path = hf_hub_download(
                        repo_id=raw_video_dataset_name,
                        filename=tarball_path,
                        repo_type="dataset",
                    )
                    print(f"  Searching for videos in {Path(tar_path).name}...")
                    with tarfile.open(tar_path, "r:gz") as tar:
                        members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]
                        print(f"  Tarball contains {len(members)} .mp4 files")
                        extracted_count = 0
                        for member in members:
                            if target_samples and len(video_files_map) >= target_samples:
                                break
                            member_name = member.name.lstrip("./")
                            tar.extract(member, extract_dir)
                            extracted_path = Path(extract_dir) / member_name
                            if extracted_path.exists():
                                vid = extracted_path.stem
                                video_files_map[vid] = str(extracted_path)
                                extracted_count += 1
                        print(f"  ✓ Extracted {extracted_count} video(s) from this tarball")
                        index_existing_mp4s()  # refresh map (handles any relative paths)
                except Exception as e:
                    print(f"  ⚠️  Could not access {tarball_path}: {e}")
                    continue

        print(f"\nTotal videos available: {len(video_files_map)}")
        if target_samples and len(video_files_map) < target_samples:
            print(f"⚠️  Only {len(video_files_map)} videos available, fewer than requested {target_samples}")

        if not video_files_map:
            print("⚠️  No videos found after attempting downloads.")
            return None

        # ------------------------------------------------------------------ #
        # Step 2: Build prompt lookup (video_id -> list of prompts)
        # ------------------------------------------------------------------ #
        print(f"\nDownloading prompts from {dataset_name}/{jsonl_path}")
        local_jsonl_path = hf_hub_download(
            repo_id=dataset_name,
            filename=jsonl_path,
            repo_type="dataset",
            cache_dir=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        )
        print(f"Loading prompts from: {local_jsonl_path}")

        prompts_map = {}  # video_id -> list of (id, text)
        with open(local_jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                video_id = data.get("video", "")
                if not video_id:
                    continue
                conversations = data.get("conversations", [])
                human_text = ""
                for conv in conversations:
                    if conv.get("from") in ("human", "user"):
                        human_text = conv.get("value", "").replace("<video>", "").strip()
                        break
                prompt_id = data.get("id", "")
                if human_text:
                    prompts_map.setdefault(video_id, []).append((prompt_id, human_text))

        print(f"Built prompt map for {len(prompts_map)} video IDs")

        # ------------------------------------------------------------------ #
        # Step 3: Construct rows: one per video/prompt (or default prompt)
        # ------------------------------------------------------------------ #
        rows = []
        default_prompt = "Describe this video in detail."
        for vid, path in video_files_map.items():
            # Try direct key, else try to match variants in prompts_map
            prompt_entries = prompts_map.get(vid, [])
            if not prompt_entries:
                # try match without Scene suffix
                base = vid.replace("v_", "").split("-Scene-")[0]
                for k, vlist in prompts_map.items():
                    if base and base in k:
                        prompt_entries = vlist
                        break
            if prompt_entries:
                for pid, text in prompt_entries:
                    rows.append(
                        {
                            "id": pid or f"{vid}_prompt",
                            "video": vid,
                            "video_url": f"file://{path}",
                            "text": text,
                        }
                    )
            else:
                rows.append(
                    {
                        "id": f"{vid}_default",
                        "video": vid,
                        "video_url": f"file://{path}",
                        "text": default_prompt,
                    }
                )

        if not rows:
            print("⚠️  No rows constructed (no videos or prompts).")
            return None

        # Limit samples if specified
        if max_samples:
            rows = rows[:max_samples]

        video_dataset = ray.data.from_items(rows)
        count = video_dataset.count()
        print(f"\n✅ Loaded {count} video QA samples from dataset")
        print("Note: Default prompt is used when no matching prompt is found in JSONL.")
        return video_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_vlm_video_config():
    """Create VLM video configuration."""
    return vLLMEngineProcessorConfig(
        model_source="Qwen/Qwen3-VL-4B-Instruct",
        # model_source="Qwen/Qwen3-VL-30B-A3B-Instruct",
        engine_kwargs=dict(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            # mm_encoder_tp_mode="data",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            gpu_memory_utilization=0.8,
            max_num_seqs=64,
        ),
        runtime_env={
            "env_vars": {"HF_TOKEN": ""},
        },
        batch_size=4,
        max_concurrent_batches=16,
        accelerator_type="L4",
        concurrency=1,
        should_continue_on_error=True,  # Skip bad videos instead of crashing
        prepare_multimodal_stage=PrepareMultimodalStageConfig(
            enabled=True,
            concurrency=(30,30),
            memory=6*1024*1024*1024,
            # model_config_kwargs=dict(
            #     # See available model config kwargs at https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.ModelConfig
            #     allowed_local_media_path="/tmp",
            # ),
        ),
        chat_template_stage=ChatTemplateStageConfig(
            enabled=True,
            concurrency=(100,100),
        ),
        tokenize_stage=TokenizerStageConfig(
            enabled=True,
            concurrency=(30,30),
        ),
        detokenize_stage=DetokenizeStageConfig(
            enabled=True,
            concurrency=(10,10),
        ),
    )


def run_vlm_video_example(max_samples: int = 1):
    """
    Run the complete VLM video QA benchmark workflow with streaming execution.
    
    Ray Data uses streaming execution by default, processing data in chunks as they're loaded.
    This enables efficient processing of large datasets without loading everything into memory.
    
    Args:
        max_samples: Maximum number of video samples to process
    
    Returns:
        Tuple of (config, processor, result, benchmark_result)
    """
    config = create_vlm_video_config()
    video_dataset = load_video_dataset(max_samples=max_samples)

    if video_dataset:
        # Build processor with preprocessing and postprocessing
        processor = build_processor(
            config, preprocess=video_preprocess, postprocess=video_postprocess
        )

        print("\n" + "=" * 60)
        print("VLM VIDEO QA BENCHMARK SETUP")
        print("=" * 60)
        print(f"Model: {config.model_source}")
        print(f"Multimodal support: {config.prepare_multimodal_stage.enabled}")
        print(f"Batch size: {config.batch_size}")
        print(f"Concurrency: {config.concurrency}")
        print(f"Streaming execution: Enabled (Ray Data default)")
        
        sample_count = video_dataset.count()
        print(f"\nProcessing {sample_count} video sample(s)...")
        
        if sample_count == 0:
            print("\n⚠️  No samples to process!")
            print("   This could mean:")
            print("   1. Videos haven't been extracted from tarballs yet")
            print("   2. Video IDs in JSONL don't match extracted video filenames")
            print("   3. Videos are in a different location than expected")
            return None, None, None, None
        
        # Measure throughput
        # Ray Data streaming execution processes data in chunks, so materialize()
        # will process the dataset in a streaming fashion
        start_time = perf_counter()
        result = processor(video_dataset).materialize()
        elapsed_time = perf_counter() - start_time
        
        # sample_count already set above
        
        # Create benchmark result
        benchmark_result = VideoCaptioningBenchmarkResult(
            samples=sample_count,
            elapsed_s=elapsed_time,
            batch_size=config.batch_size,
            concurrency=config.concurrency,
        )
        
        # Display results
        benchmark_result.show()
        
        # Show results
        sample_results = result.take(min(5, sample_count))
        print(f"\n{'=' * 60}")
        print(f"RESULTS ({len(sample_results)} sample(s)):")
        print("=" * 60)
        for i, res in enumerate(sample_results, 1):
            print(f"\n  Sample {i}:")
            print(f"    ID: {res.get('id', 'N/A')}")
            print(f"    Video: {res.get('video', 'N/A')}")
            # Get the original prompt text from the dataset
            prompt_text = res.get('text', 'N/A')
            print(f"    Prompt: {prompt_text[:150]}{'...' if len(str(prompt_text)) > 150 else ''}")
            response = res.get('caption', res.get('resp', 'N/A'))
            print(f"    Response: {response[:200]}{'...' if len(str(response)) > 200 else ''}")
        
        return config, processor, result, benchmark_result
    return None, None, None, None


if __name__ == "__main__":
    # Run a minimal video QA benchmark example (1 sample) to prove it works
    try:
        import torch

        if torch.cuda.is_available():
            # Initialize Ray
            ray.init()
            try:
                print("=" * 60)
                print("MINIMAL VIDEO QA BENCHMARK (1 sample)")
                print("=" * 60)
                # Run benchmark with 500 samples (may take time/download multiple tarballs)
                config, processor, result, benchmark_result = run_vlm_video_example(max_samples=100)
                
                if benchmark_result:
                    print("\n✅ Benchmark completed successfully!")
                    print(f"   Processed {benchmark_result.samples} sample(s)")
                    print(f"   Throughput: {benchmark_result.throughput:.2f} videos/s")
                else:
                    print("\n⚠️  Benchmark did not complete - check logs above for details")
            finally:
                ray.shutdown()
        else:
            print("Skipping VLM video QA benchmark (no GPU available)")
    except Exception as e:
        print(f"Skipping VLM video QA benchmark due to environment error: {e}")
        import traceback
        traceback.print_exc()
