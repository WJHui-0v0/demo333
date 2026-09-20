import argparse
import gc
import json
import os
import shutil

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import swanlab
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler

from Dataset import bc2gmDataset
from model import Qwen4NER
from prompt_utils import get_generation_eos_ids
from utils import Arguments, BestModelSaver, Metrics, get_next, write_log


class Trainer:
    """Custom training loop with memory behavior similar to HF Trainer."""

    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config
        self.num_epochs = config.num_epochs
        self.grad_accum_steps = max(
            1, int(getattr(config, "gradient_accumulation_steps", 1))
        )
        self.accelerator = Accelerator(
            mixed_precision="bf16" if getattr(config, "bf16", True) else "fp16"
        )
        self.device = self.accelerator.device
        self.metrics = Metrics(config)
        self.save_dir = get_next(config.save_dir)
        self.best_saver = BestModelSaver(
            save_dir=self.save_dir,
            delta=getattr(config, "delta", 1e-8),
        )
        self.log_dir = os.path.join(self.save_dir, "log.jsonl")
        self.optimizer = None
        self.scheduler = None
        self.model = None

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        print(config.get_args_dict())

    def reset_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_memory_stats(self):
        if not torch.cuda.is_available():
            return {key: 0.0 for key in (
                "peak_allocated_mib", "peak_reserved_mib",
                "current_allocated_mib", "current_reserved_mib",
            )}
        torch.cuda.synchronize(self.device)
        return {
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(self.device) / 2**20, 2
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(self.device) / 2**20, 2
            ),
            "current_allocated_mib": round(
                torch.cuda.memory_allocated(self.device) / 2**20, 2
            ),
            "current_reserved_mib": round(
                torch.cuda.memory_reserved(self.device) / 2**20, 2
            ),
        }

    def log_memory_stats(self, stage, epoch=None):
        stats = self.get_memory_stats()
        print(
            f"[{stage}] peak allocated={stats['peak_allocated_mib']:.2f} MiB, "
            f"peak reserved={stats['peak_reserved_mib']:.2f} MiB, "
            f"current allocated={stats['current_allocated_mib']:.2f} MiB"
        )
        log_data = {f"{stage}/{key}": value for key, value in stats.items()}
        if epoch is not None:
            log_data["epoch"] = epoch + 1
        swanlab.log(log_data)
        return stats

    def _enable_training_memory_features(self):
        if getattr(self.config, "gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False

    @property
    def global_step(self):
        return getattr(self, "_global_step", 0)

    @global_step.setter
    def global_step(self, value):
        self._global_step = int(value)

    def _load_checkpoint(self, checkpoint_dir):
        state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
            self.global_step = state.get("global_step", 0)
            self.start_epoch = state.get("epoch", 0)
            self.log_history = state.get("log_history", [])
        else:
            self.start_epoch = 0
            self.log_history = []

        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )
        if os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=self.device)
            )

        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            from safetensors.torch import load_file
            state_dict = load_file(adapter_path, device=str(self.device))
            self.model.load_state_dict(state_dict, strict=False)
        print(
            f"Resumed checkpoint {checkpoint_dir}: "
            f"global_step={self.global_step}, epoch={getattr(self, 'start_epoch', 0)}"
        )

    def _save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(
            self.save_dir, f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.accelerator.unwrap_model(self.model).save_pretrained(checkpoint_dir)
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(checkpoint_dir, "scheduler.pt"),
        )
        with open(
            os.path.join(checkpoint_dir, "trainer_state.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "global_step": self.global_step,
                    "epoch": epoch,
                    "log_history": getattr(self, "log_history", []),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        limit = int(getattr(self.config, "save_total_limit", 3))
        checkpoints = []
        for name in os.listdir(self.save_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(self.save_dir, name)
                if os.path.isdir(path):
                    try:
                        checkpoints.append((int(name.rsplit("-", 1)[1]), path))
                    except ValueError:
                        pass
        checkpoints.sort()
        for _, path in checkpoints[:-limit]:
            shutil.rmtree(path, ignore_errors=True)
        print(f"Saved checkpoint: {checkpoint_dir}")

    def _sample_train_eval(self, sample_batches):
        self.model.eval()
        self.metrics.reset()
        with torch.inference_mode():
            for original_batch in sample_batches:
                batch = {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in original_batch.items()
                }
                batch["input_ids"] = batch["input_ids"].to(
                    self.device, non_blocking=True
                )
                batch["attention_mask"] = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                with self.accelerator.autocast():
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=True,
                        max_new_tokens=int(
                            getattr(self.config, "eval_max_new_tokens", 128)
                        ),
                        pad_token_id=self.model.config.pad_token_id,
                        eos_token_id=get_generation_eos_ids(self.config.tokenizer),
                        do_sample=False,
                    )
                prompt_width = batch["input_ids"].shape[1]
                responses = self.config.tokenizer.batch_decode(
                    generated_ids[:, prompt_width:],
                    skip_special_tokens=True,
                )
                predictions = [self.metrics.parse_json(text) for text in responses]
                self.metrics.add_entities(
                    predictions, batch["entities"], batch["text"]
                )
                del generated_ids, responses, predictions
        result = self.metrics.get_result_dict()
        self.metrics.reset()
        self.model.train()
        return result

    def train(self, train_loader, dev_loader, test_loader, model, optimizer):
        updates_per_epoch = (
            len(train_loader) + self.grad_accum_steps - 1
        ) // self.grad_accum_steps
        total_updates = self.num_epochs * updates_per_epoch
        warmup_steps = int(
            total_updates * float(getattr(self.config, "warmup_ratio", 0.0))
        )
        if warmup_steps <= 0:
            warmup_steps = min(int(self.config.warmup_steps), total_updates)
        self.scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
        (
            self.model,
            self.optimizer,
            self.scheduler,
            train_loader,
            dev_loader,
            test_loader,
        ) = self.accelerator.prepare(
            model, optimizer, self.scheduler, train_loader, dev_loader, test_loader
        )
        self._enable_training_memory_features()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        write_log(self.log_dir, {"config": self.config.get_args_dict()})
        self.global_step = 0
        self.start_epoch = 0
        self.log_history = []
        resume_path = getattr(self.config, "resume_from_checkpoint", None)
        if resume_path:
            self._load_checkpoint(resume_path)
        sample_limit = max(0, int(getattr(self.config, "train_eval_batches", 2)))
        sample_batches = []
        if sample_limit:

            sample_iterator = iter(dev_loader)
            for _ in range(sample_limit):
                try:
                    sample_batches.append(next(sample_iterator))
                except StopIteration:
                    break

        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            self.model.config.use_cache = False
            self.reset_peak_memory()
            self.optimizer.zero_grad(set_to_none=True)
            total_train_loss = 0.0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]"
            )

            for step, batch in enumerate(progress_bar):
                if (
                    epoch == self.start_epoch
                    and self.global_step > 0
                    and step < self.global_step % len(train_loader)
                ):
                    continue
                with self.accelerator.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        return_dict=False,
                        use_cache=False,
                    )
                    raw_loss = outputs[0]
                    loss = raw_loss / self.grad_accum_steps

                self.accelerator.backward(loss)
                total_train_loss += raw_loss.detach().float().item()
                should_step = (
                    (step + 1) % self.grad_accum_steps == 0
                    or step + 1 == len(train_loader)
                )
                if should_step:
                    if getattr(self.config, "max_grad_norm", 0) > 0:
                        self.accelerator.clip_grad_norm_(
                            trainable_params, self.config.max_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                loss_value = raw_loss.detach().float().item()
                progress_bar.set_postfix(loss=f"{loss_value:.4f}")
                if step % max(1, int(getattr(self.config, "log_steps", 10))) == 0:
                    allocated = (
                        torch.cuda.memory_allocated(self.device) / 2**20
                        if torch.cuda.is_available() else 0.0
                    )
                    swanlab.log({
                        "train/loss_step": loss_value,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/current_allocated_mib": allocated,
                    })
                eval_steps = int(getattr(self.config, "eval_steps", 0))
                if should_step and eval_steps > 0 and self.global_step % eval_steps == 0:
                    sampled = self._sample_train_eval(sample_batches)
                    swanlab.log({
                        "train_sample/f1": sampled["micro_avg"]["f1"],
                        "train_sample/precision": sampled["micro_avg"]["precision"],
                        "train_sample/recall": sampled["micro_avg"]["recall"],
                        "step": self.global_step,
                    })
                save_steps = int(getattr(self.config, "save_steps", 0))
                if should_step and save_steps > 0 and self.global_step % save_steps == 0:
                    self._save_checkpoint(epoch)

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            self.log_memory_stats("train", epoch)
            swanlab.log({"train/loss_epoch": avg_train_loss})

            results = self.eval(epoch, dev_loader, is_test=False)
            f1 = results["micro_avg"]["f1"]
            write_log(self.log_dir, {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "eval/f1": f1,
                "eval/results": results,
            })
            self.best_saver.update(
                f1=f1,
                model=self.accelerator.unwrap_model(self.model),
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._save_checkpoint(self.num_epochs)
        return self.best_saver.get_best_lora_path()

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
        self.model.config.use_cache = True
        self.reset_peak_memory()
        stage = "test" if is_test else "eval"
        desc = "Testing" if is_test else "Evaluation"
        eos_token_id = get_generation_eos_ids(self.config.tokenizer)
        max_new_tokens = int(
            getattr(
                self.config,
                "test_max_new_tokens" if is_test else "eval_max_new_tokens",
                self.config.max_new_tokens,
            )
        )

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc=desc):
                batch["input_ids"] = batch["input_ids"].to(
                    self.device, non_blocking=True
                )
                batch["attention_mask"] = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                with self.accelerator.autocast():
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.model.config.pad_token_id,
                        eos_token_id=eos_token_id,
                        do_sample=False,
                    )
                prompt_width = batch["input_ids"].shape[1]
                responses = self.config.tokenizer.batch_decode(
                    generated_ids[:, prompt_width:],
                    skip_special_tokens=True,
                )
                predictions = [self.metrics.parse_json(text) for text in responses]
                self.metrics.add_entities(
                    predictions, batch["entities"], batch["text"]
                )
                del generated_ids, responses, predictions

        self.log_memory_stats(stage, epoch)
        results = self.metrics.get_result_dict()
        self.metrics.reset()
        print(
            f"{desc} micro-F1: {results['micro_avg']['f1']:.4f} | "
            f"macro-F1: {results['macro_avg']['f1']:.4f}"
        )
        swanlab.log({
            f"{stage}/f1": results["micro_avg"]["f1"],
            f"{stage}/epoch": epoch,
        })
        write_log(self.log_dir, {
            f"{stage}/results": results,
            f"{stage}/epoch": epoch,
        })
        if is_test:
            self.save_model()
        return results

    def save_model(self):
        save_path = os.path.join(self.save_dir, "best_lora_final")
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(save_path)
        self.config.tokenizer.save_pretrained(save_path)
        print(f"LoRA adapter saved to {save_path}")


def build_subset(dataset, size):
    if size <= 0 or size >= len(dataset):
        return
    dataset.texts = dataset.texts[:size]
    dataset.label_list = dataset.label_list[:size]
    dataset.samples = []
    dataset._pretokenize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg", type=str, required=True)
    parser.add_argument("--quick_test", action="store_true")
    args_cli = parser.parse_args()
    args = Arguments(args_cli.arg)

    train_dataset = bc2gmDataset(args, args.train_path, is_train=True)
    dev_dataset = bc2gmDataset(args, args.dev_path, is_train=False)
    test_dataset = bc2gmDataset(args, args.test_path, is_train=False)
    if args_cli.quick_test:
        build_subset(train_dataset, int(getattr(args, "quick_train_size", 16)))
        build_subset(dev_dataset, int(getattr(args, "quick_dev_size", 8)))
        build_subset(test_dataset, int(getattr(args, "quick_test_size", 8)))
        args.num_epochs = 1
        print(
            f"Quick test: train={len(train_dataset)}, "
            f"dev={len(dev_dataset)}, test={len(test_dataset)}"
        )

    train_loader = train_dataset.get_data_loader(
        batch_size=args.micro_batch_size, shuffle=True
    )
    eval_batch_size = int(getattr(args, "eval_batch_size", 1))
    dev_loader = dev_dataset.get_data_loader(
        batch_size=eval_batch_size, shuffle=False
    )
    test_loader = test_dataset.get_data_loader(
        batch_size=eval_batch_size, shuffle=False
    )

    model_config = Qwen4NER(args)
    model = model_config.get_model()
    optimizer = model_config.get_optimizer()
    swanlab.init(
        project="qwen4ner",
        name=f"{args.model_name}-{args.method}",
        config=args.get_args_dict(),
    )
    trainer = Trainer(args, model_config)
    best_path = trainer.train(
        train_loader, dev_loader, test_loader, model, optimizer
    )

    trainer.model, trainer.optimizer, trainer.scheduler = (
        trainer.accelerator.free_memory(
            trainer.model, trainer.optimizer, trainer.scheduler
        )
    )
    del model, optimizer
    model_config._release_models()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.model = model_config.load_adapter(best_path).to(
        model_config.compute_dtype
    )
    trainer.model = trainer.model.to(trainer.device)
    trainer.eval(epoch=0, data_loader=test_loader, is_test=True)
    swanlab.finish()
