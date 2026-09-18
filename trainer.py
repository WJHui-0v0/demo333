import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import swanlab
import torch
import gc

from tqdm import tqdm
from model import Qwen4NER
from transformers import get_scheduler
from Datase import bc2gmDataset
from utils import get_next, write_log, BestModelSaver, Arguments, Metrics

from accelerate import Accelerator
from prompt_utils import get_generation_eos_ids

class Trainer:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config
        self.num_epochs = config.num_epochs

        self.accelerator = Accelerator(mixed_precision="bf16")
        self.device = self.accelerator.device
        self.metrics = Metrics(config)
        self.save_dir = get_next(config.save_dir)

        self.best_saver = BestModelSaver(
            save_dir=self.save_dir,
            delta=getattr(config, "delta", 1e-8),
        )

        self.log_dir = os.path.join(self.save_dir,"log.jsonl",)

        self.optimizer = None
        self.scheduler = None
        self.model = None

        print(config.get_args_dict())
        
    def reset_peak_memory(self):
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            self.device
        )  
        
    def get_memory_stats(self):
        if not torch.cuda.is_available():
            return {
                "peak_allocated_mib": 0.0,
                "peak_reserved_mib": 0.0,
                "current_allocated_mib": 0.0,
                "current_reserved_mib": 0.0,
            }

        torch.cuda.synchronize(self.device)

        return {
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(
                    self.device
                ) / 1024**2,
                2,
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(
                    self.device
                ) / 1024**2,
                2,
            ),
            "current_allocated_mib": round(
                torch.cuda.memory_allocated(
                    self.device
                ) / 1024**2,
                2,
            ),
            "current_reserved_mib": round(
                torch.cuda.memory_reserved(
                    self.device
                ) / 1024**2,
                2,
            ),
        }
    def log_memory_stats(self, stage, epoch=None):
        stats = self.get_memory_stats()
        
        print(
            f"[{stage}] "
            f"peak allocated: "
            f"{stats['peak_allocated_mib']:.2f} MiB, "
            f"peak reserved: "
            f"{stats['peak_reserved_mib']:.2f} MiB, "
            f"current allocated: "
            f"{stats['current_allocated_mib']:.2f} MiB"
        )

        log_data = {
            f"{stage}/peak_allocated_mib": (
                stats["peak_allocated_mib"]
            ),
            f"{stage}/peak_reserved_mib": (
                stats["peak_reserved_mib"]
            ),
            f"{stage}/current_allocated_mib": (
                stats["current_allocated_mib"]
            ),
            f"{stage}/current_reserved_mib": (
                stats["current_reserved_mib"]
            ),
        }

        if epoch is not None:
            log_data["epoch"] = epoch + 1

        swanlab.log(log_data)

        return stats
    def train(self,traindataLoader, devdataLoader, testdataLoader, model,optimizer):
        
        self.device = self.accelerator.device
                
        scheduler=get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.num_epochs * len(traindataLoader)
        )


        self.model, self.optimizer, self.scheduler, traindataLoader, devdataLoader, testdataLoader=self.accelerator.prepare(model, optimizer, scheduler, traindataLoader, devdataLoader, testdataLoader)

        trainable_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]
        #梯度检查点
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        
        write_log(self.log_dir, {"config": self.config.get_args_dict()})
        
        for epoch in range(self.num_epochs):
            self.model.train()
            self.model.config.use_cache = False
            
            # torch.cuda.empty_cache()
            self.reset_peak_memory()
            
            total_train_loss = 0
            progress_bar = tqdm(
                traindataLoader, 
                desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]",
            )
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                self.optimizer.zero_grad()
                
        
                with self.accelerator.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=False,
                        use_cache=False,
                    )

                    loss = outputs[0]
                self.accelerator.backward(loss)
                
                # self.accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                    total_train_loss += loss.item()
                    loss_record = loss.item()
                
                progress_bar.set_postfix({"Loss": loss_record})
                
                current_allocated_mib  = torch.cuda.memory_allocated(self.accelerator.device)/1024**2

                if step % 10 == 0:
                    swanlab.log({
                        "train/loss_step": loss_record,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "train/current_allocated_mib": (current_allocated_mib)
                    })
                    
            avg_train_loss = total_train_loss / len(traindataLoader)
            self.log_memory_stats(
                stage="train",
                epoch=epoch,
            )
            swanlab.log({ 
                "train/loss_epoch": avg_train_loss
            })
            
            results_dict = self.eval(epoch, devdataLoader, is_test=False)
            
            torch.cuda.empty_cache()
            gc.collect()

            f1 = results_dict['micro_avg']['f1']
            
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "eval/f1": results_dict['micro_avg']['f1'],
                "eval/results": results_dict
            }
            write_log(self.log_dir, log_dict)

            model_to_save = (
                self.accelerator.unwrap_model(
                    self.model
                )
            )

            self.best_saver.update(
                f1=f1,
                model=model_to_save,
            )
        
        return self.best_saver.get_best_lora_path()    
    
        
    def eval(self,epoch, dataLoader,is_test=False):

        self.model.eval()
        self.model.config.use_cache = True
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        
        stage = "test" if is_test else "eval"    
        desc = "Evaluation" if not is_test else "Testing"
        
        progress_bar = tqdm(dataLoader, desc=desc)
        
        eos_token_id = get_generation_eos_ids(self.config.tokenizer)
        
        with torch.inference_mode():
            for step, batch in enumerate(progress_bar):
                input_ids=batch["input_ids"].to(self.device)
                attention_mask=batch["attention_mask"].to(self.device)
                texts=batch["text"]
                entities=batch["entities"]
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # with self.accelerator.autocast():
                    generated_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.model.config.pad_token_id,
                    eos_token_id=eos_token_id,
                    do_sample=False
                )
                    
                prompt_width = input_ids.shape[1]

                generated_only = generated_ids[:, prompt_width:]  
                
                response = self.config.tokenizer.batch_decode(generated_only, skip_special_tokens=True)
                # if step == 0:
                #     print("====模型生成输出====")
                #     print(response[0])
                
                pred_entities_batch = [self.metrics.parse_json(r) for r in response]
                self.metrics.add_entities(pred_entities_batch, entities, texts)
                del generated_ids, generated_only, response, pred_entities_batch

        
        self.log_memory_stats(
            stage=stage,
            epoch=epoch,
        ) 
               
        results_dict = self.metrics.get_result_dict()
        
        epoch_tag = epoch + 1 if epoch is not None else "-"
        print(f"\n========== {desc} 指标 (epoch {epoch_tag}) ==========")
        print(self.metrics.result_df.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"{desc} micro-F1: {results_dict['micro_avg']['f1']:.4f} | "
              f"macro-F1: {results_dict['macro_avg']['f1']:.4f}")
        
        self.metrics.reset()
        
        micro_f1 = results_dict["micro_avg"]["f1"]
        print(f"{desc} F1 Score: {micro_f1:.4f}")

        swanlab.log({
            f"{stage}/f1": micro_f1,
            f"{stage}/epoch": epoch,
        })

        write_log(self.log_dir, {
            f"{stage}/results": results_dict,
            f"{stage}/epoch": epoch,
        })        
        
        if is_test:
            self.save_model()
        return results_dict
    
    def save_model(self):
        save_path = os.path.join(self.save_dir,"best_lora_final", )

        model = self.model
        if getattr(self, "accelerator", None) is not None:
            model = self.accelerator.unwrap_model(self.model)
        
        model.save_pretrained(save_path)
        self.config.tokenizer.save_pretrained(save_path)

        print(f"LoRA adapter saved to {save_path}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arg',
        type=str,
        default='./args/arg1.json'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='快速测试模式'
    )
    cmd_args = parser.parse_args()
    args=Arguments(cmd_args.arg)
    
    train_dataset = bc2gmDataset(args, args.train_path, is_train=True)
    dev_dataset = bc2gmDataset(args, args.dev_path, is_train=False)
    test_dataset = bc2gmDataset(args, args.test_path, is_train=False)
    
    if cmd_args.quick_test:
        train_dataset.texts = train_dataset.texts[:625]
        train_dataset.label_list = train_dataset.label_list[:625]
        dev_dataset.texts = dev_dataset.texts[:125]
        dev_dataset.label_list = dev_dataset.label_list[:125]
        test_dataset.texts = test_dataset.texts[:250]
        test_dataset.label_list = test_dataset.label_list[:250]
        args.num_epochs = 1
        print("\n========== 快速测试模式 ==========")
        print(f"训练集：{len(train_dataset)} 条")
        print(f"验证集：{len(dev_dataset)} 条")
        print(f"测试集：{len(test_dataset)} 条")
        print(f"训练轮数：{args.num_epochs}")
        print("==================================\n")
        
    dev_dataloader = dev_dataset.get_data_loader(batch_size=args.batch_size * 4, shuffle=False)
    test_dataloader = test_dataset.get_data_loader(batch_size=args.batch_size * 4, shuffle=False)
    train_dataloader = train_dataset.get_data_loader(batch_size=args.batch_size, shuffle=True)

    model4ner=Qwen4NER(args)
    model=model4ner.get_model()
    optimizer = model4ner.get_optimizer()
    
    swanlab.init(
        project="qwen4ner",
        name=f"{args.model_name}-{args.method}-{args.lora_target_modules}",
        config=args.get_args_dict()
    )
    
    trainer=Trainer(args, model4ner)
    best_lora_path = trainer.train(
                        train_dataloader,
                        dev_dataloader,
                        test_dataloader,
                        model,
                        optimizer,
                    )
    trainer.model = None
    trainer.optimizer = None
    trainer.scheduler = None

    del model
    del optimizer

    trainer.model_config._release_models()
    del trainer.accelerator
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    trainer.model = trainer.model_config.load_adapter(best_lora_path) 
    trainer.model = trainer.model.to(torch.bfloat16)   # ← 权重压成 bf16（核心修复）
    trainer.model = trainer.model.to(trainer.device)
    
    trainer.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    trainer.model.eval()
    
    print("[test] model dtype =", next(trainer.model.parameters()).dtype)
    
    trainer.eval(
        epoch=0,
        dataLoader=test_dataloader,
        is_test=True,
    )    
        
        
    swanlab.finish()    