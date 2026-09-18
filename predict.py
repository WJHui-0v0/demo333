from model import Qwen4NER
from transformers import AutoTokenizer
from utils import Arguments
import argparse
import torch
from prompt_utils import build_prompt_text, get_generation_eos_ids

class Predictor:
    def __init__(self, config: Arguments, weight: str):
        self.config = config
        self.tokenizer = config.tokenizer
        self.tokenizer.padding_side = "left"
        
        self.model_config = Qwen4NER(config,skip_load=True)
        self.model = self.model_config.load_adapter(weight)
        self.model = self.model.to(config.device)
        self.model.eval()
        
    def predict(self, text):
        prompt_text = build_prompt_text(
            self.tokenizer,
            text,
            self.config.prompt,
        )

        inputs = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.inference_mode():
            eos_token_id = get_generation_eos_ids(
                self.tokenizer
            )

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        # 只取新生成的部分
        generated_only = generated_ids[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=str, default='./args/arg1.json')
    parser.add_argument('--weight', type=str, default='./checkpoint/exp1/best_model')
    parser.add_argument('--text', type=str, default='Using the same approach we have shown that hFIRE binds the stimulatory proteins Sp1 and Sp3 in addition to CBF')
    
    args = parser.parse_args()
    
    config=Arguments(args.arg)
    weight=args.weight
    predictor=Predictor(config, weight)
    response=predictor.predict(args.text)
    print(response)
        
        
        
        

    

        
        