import gc

import torch
import torch.nn as nn
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from utils import Arguments

class Qwen4NER(nn.Module):
    def __init__(self, config,skip_load=False):
        super().__init__()
        
        self.config = config
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        
        self.model = None
        self.base_model = None
        if not skip_load:
            self.set_model()
     
     
    def _build_lora_config(self):
        """ LoRA """
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )       
    
    def _build_bnb_config(self):
        """4bit 量化"""
        if self.config.method != "qlora":
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )    
         
    def _load_base_model(self):
            
        bnb_config = self._build_bnb_config()
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_dir,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if bnb_config is not None:
            # QLoRA 使用 4bit 量化模型
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # 普通 LoRA 使用 bf16
            load_kwargs["torch_dtype"] = torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        base_model.config.pad_token_id = (self.config.tokenizer.pad_token_id)
        base_model.config.eos_token_id = (self.config.tokenizer.eos_token_id)

        return base_model
    
    def set_model(self):
  
        self.base_model = self._load_base_model()

        if self.config.method == "qlora":
            self.base_model = prepare_model_for_kbit_training(
                self.base_model
            )

        if self.config.method in ("lora", "qlora"):
            self.model = get_peft_model(
                self.base_model,
                self._build_lora_config(),
            )

            self.model.print_trainable_parameters()

        else:
            self.model = self.base_model

        self.model.config.use_cache = False

    def get_optimizer(self):

        trainable_params = [
            param
            for param in self.model.parameters()
            if param.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError(
                "没有找到可训练参数，请检查 LoRA 配置"
            )

        if self.config.method == "qlora":
            return bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        return torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _release_models(self):

        self.model = None
        self.base_model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_adapter(self, save_path):
       
        self._release_models()

        self.base_model = self._load_base_model()

        self.base_model.config.use_cache = True

        self.model = PeftModel.from_pretrained(
            self.base_model,
            save_path,
            is_trainable=False,
        )

        self.model.eval()

        return self.model

    def load_model(self, model_path):
        
        self._release_models()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.model.config.pad_token_id = (self.config.tokenizer.pad_token_id)
        self.model.config.eos_token_id = (self.config.tokenizer.eos_token_id)

        self.model.config.use_cache = True
        self.model.eval()

        return self.model

    def get_model(self):
        return self.model