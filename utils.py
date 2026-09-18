import os, re, json
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, PreTrainedModel
from peft import PeftModel
from Dataset import bc2gmDataset
import torch
import numpy as np
from collections import Counter
import shutil
from accelerate import Accelerator
from peft import PeftModel
from collections import defaultdict


def get_next(prefix_dir):
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir+'/exp1')
        return prefix_dir+'/exp1'
    else:
        existing_nums = []
        for file in os.listdir(prefix_dir):
            if file.startswith('exp'):
                existing_nums.append(int(file[3:]))
        if len(existing_nums) == 0:
            next_num = 1
        else:        
            next_num = max(existing_nums) + 1
        os.makedirs(prefix_dir+'/exp'+str(next_num))
        return prefix_dir+'/exp'+str(next_num)




def write_log(log_jsonl_path, log_dict):

    with open(log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")


class Metrics:
    def __init__(self, config, eps=1e-8):
        self.eps = eps
        with open(os.path.join(config.data_path, 'labels.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entity_types = sorted(set(data))
        self.reset()

    @staticmethod
    def _normalize(text):
        return re.sub(r'\s+', ' ', str(text)).strip() if text else ""

    def parse_json(self, json_str):
        if not json_str or not json_str.strip():
            return []
        s = json_str.strip()
        s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
        s = re.sub(r'\s*```\s*$', '', s)
        try:
            data = json.loads(s)
            ents = self._extract_from_json(data)
            if ents is not None:
                return ents
        except Exception:
            pass
        return self._extract_from_text(s)

    def _extract_from_json(self, data):
        if isinstance(data, dict):
            raw = data.get('entities') or data.get('entity_list') or data.get('mentions')
            if raw is None:
                return None
        elif isinstance(data, list):
            raw = data
        else:
            return None
        out = []
        for e in raw:
            if isinstance(e, dict):
                typ = e.get('type') or e.get('entity_type') or 'GENE'
                name = e.get('name') or e.get('entity') or e.get('mention') or e.get('text')
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                name, typ = e[0], e[1]
            elif isinstance(e, str):
                name, typ = e, 'GENE'
            else:
                continue
            if name:
                out.append({'type': typ, 'name': str(name)})
        return out

    def _extract_from_text(self, s):
        if s.strip() in ('无实体', 'None', 'null', ''):
            return []
        out = []
        for line in s.split('\n'):
            line = line.strip()
            if not line or line == '无实体':
                continue
            if ':' in line:
                name, typ = line.rsplit(':', 1)
                out.append({'type': typ.strip().upper(), 'name': name.strip()})
            else:
                out.append({'type': 'GENE', 'name': line})
        return out

    def add_entities(self, batch_predictions, batch_labels, batch_texts):
        for preds, labels, texts in zip(batch_predictions, batch_labels, batch_texts):
            true_set = set()
            for te in labels:
                typ = te.get('type'); pos = te.get('pos')
                if typ in self.entity_types and pos is not None:
                    gold_text = texts[int(pos[0]):int(pos[1])]
                    true_set.add((self._normalize(gold_text), typ))
            pred_set = set()
            for pe in preds:
                if not isinstance(pe, dict):
                    continue
                typ = pe.get('type')
                name = pe.get('name') or pe.get('entity') or pe.get('mention')
                if not name or not typ or typ not in self.entity_types:
                    continue
                pred_set.add((self._normalize(str(name)), typ))
            # 逐样本求交集，再跨样本累加（避免跨样本污染）
            types = (set(self.entity_types)
                     | {t for _, t in true_set} | {t for _, t in pred_set})
            for typ in types:
                t = {x for x in true_set if x[1] == typ}
                p = {x for x in pred_set if x[1] == typ}
                self.tp_dict[typ]        += len(p & t)
                self.pred_sum_dict[typ]  += len(p)
                self.true_sum_dict[typ]  += len(t)

    def reset(self):
        self.tp_dict = defaultdict(int)
        self.pred_sum_dict = defaultdict(int)
        self.true_sum_dict = defaultdict(int)
        self.result_df = None

    def get_results(self):
        all_types = sorted(set(self.tp_dict) | set(self.pred_sum_dict) | set(self.true_sum_dict))
        results = []
        for typ in all_types:
            tp = self.tp_dict.get(typ, 0); pre = self.pred_sum_dict.get(typ, 0); tru = self.true_sum_dict.get(typ, 0)
            p = tp / (pre + self.eps); r = tp / (tru + self.eps)
            f1 = 2 * p * r / (p + r + self.eps)
            results.append({'precision': p, 'recall': r, 'f1': f1, 'support': tru})
        df = pd.DataFrame(results, index=all_types)
        if not df.empty:
            df.loc['macro_avg'] = df[['precision', 'recall', 'f1']].mean()
            df.loc['macro_avg', 'support'] = float('nan')
        total_tp = sum(self.tp_dict.values()); total_pre = sum(self.pred_sum_dict.values()); total_tru = sum(self.true_sum_dict.values())
        mp = total_tp / (total_pre + self.eps); mr = total_tp / (total_tru + self.eps)
        mf1 = 2 * mp * mr / (mp + mr + self.eps)
        df.loc['micro_avg'] = [mp, mr, mf1, float('nan')]
        self.result_df = df
        return df

    def get_result_dict(self):
        if self.result_df is None:
            df = self.get_results()
        else:
            df = self.result_df
        result = {}
        for idx in df.index:
            key = str(idx); row = df.loc[idx]; row_dict = {}
            for col in df.columns:
                val = row[col]
                row_dict[col] = None if pd.isna(val) else (val.item() if isinstance(val, (np.integer, np.floating)) else val)
            result[key] = row_dict
        return result



class BestModelSaver:
    
    def __init__(self, save_dir, delta=1e-8):
        self.save_dir = save_dir
        self.best_f1 = -1.0
        self.best_model_path = None
        self.delta = delta

    def update(self, f1, model):
        
        if f1 > self.best_f1 + self.delta:
            self.best_f1 = f1
            save_path = os.path.join(self.save_dir, "best_lora")
            model.save_pretrained(save_path)
            self.best_model_path = save_path
            print(f"New best f1:{f1:.4f}, save lora to {save_path}")

    def get_best_lora_path(self):
        if self.best_model_path is None:
            raise RuntimeError("尚未保存任何最优lora，检查dev f1是否有效")
        return self.best_model_path

class Arguments:
    def __init__(self, config_path="arguments.json"):
        self.args_dict = self._load_json_config(config_path)
        self.__dict__.update(self.args_dict)
        self._set_seed()
        self._set_tokenizer()
     
    def _set_seed(self):
        seed = self.args_dict.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _set_tokenizer(self):
        model_dir = self.args_dict.get("model_dir")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

    def _load_json_config(self, config_path):
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_args_dict(self):
        return self.args_dict
    