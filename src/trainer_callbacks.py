from transformers import TrainerCallback, GenerationConfig, TrainerControl, TrainerState
import torch
from tqdm import tqdm
import csv
import os
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule,
)
from torch.utils.tensorboard import SummaryWriter


class LLMSampleCB(TrainerCallback):
    def __init__(
        self,
        trainer,
        test_dataset,
        max_new_tokens=256,
        log_dir="samples",
    ):
        """A Callback to log samples to CSV during training"""
        super().__init__()
        self.sample_dataset = test_dataset
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path, max_new_tokens=max_new_tokens
        )
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def generate(self, messages):
        # Apply chat template to convert messages to prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenize with attention mask
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.model.device)
        attention_mask = tokenized["attention_mask"].to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.gen_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][len(input_ids[0]) :], skip_special_tokens=True
        )

    def log_samples(self, examples, step):
        """Generate and log samples to CSV"""
        csv_path = os.path.join(self.log_dir, f"samples_step_{step}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header with generation config
            writer.writerow(
                ["prompt", "generation"] + list(self.gen_config.to_dict().keys())
            )

            for example in tqdm(examples, leave=False):
                messages = example["messages"][:-1]
                generation = self.generate(messages=messages)
                # Convert messages to readable prompt for CSV
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                writer.writerow(
                    [prompt, generation] + list(self.gen_config.to_dict().values())
                )

        print(f"Samples saved to: {csv_path}")

    def on_evaluate(self, args, state, control, **kwargs):
        """Log samples after calling trainer.evaluate"""
        try:
            print(
                f"\n=== LLMSampleCB: Starting to log samples at step {state.global_step} ==="
            )
            self.log_samples(self.sample_dataset, state.global_step)
        except Exception as e:
            print(f"\n!!! ERROR in LLMSampleCB: {type(e).__name__}: {e} !!!")
            import traceback

            traceback.print_exc()


class ProfilerCallback(TrainerCallback):
    def __init__(self, log_dir="./logs/train/profiler", activities=None):
        self.log_dir = log_dir
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.prof = None

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        # Schedule: wait 1 step, warmup 1 step, actively profile 3 steps
        self.prof = profile(
            activities=self.activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1000),
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.prof.__enter__()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.prof:
            self.prof.step()  # flush profiler per step

    def on_train_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.prof:
            self.prof.__exit__(None, None, None)


class CustomTensorBoardCallback(TrainerCallback):
    """
    Custom TensorBoard callback that opens and closes the writer for each logging event.
    This prevents file handles from remaining open throughout the entire training run.

    Unlike the default TensorBoard integration which keeps a writer open the whole time,
    this callback creates a writer, logs metrics, then immediately closes it.
    """

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory where TensorBoard logs will be written
        """
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        """
        Log metrics to TensorBoard, opening and closing the writer each time.

        This is called whenever the trainer logs metrics (controlled by logging_steps).
        """
        if not state.is_world_process_zero:
            return

        if logs is None:
            return

        # Create a new writer, log metrics, then close it
        writer = SummaryWriter(log_dir=self.log_dir)

        try:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{key}", value, state.global_step)
                # Could add support for other types here (images, histograms, etc.)

            # Ensure everything is written to disk
            writer.flush()
        finally:
            # Always close the writer to release file handles
            writer.close()
