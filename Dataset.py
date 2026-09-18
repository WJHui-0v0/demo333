from __future__ import annotations
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import json
from prompt_utils import build_prompt_text, build_sft_text, build_target

class bc2gmDataset(Dataset):
    def __init__(self, args, data_path, is_train):

        self.is_train = is_train
        self.tokenizer = args.tokenizer
        self.max_length = args.max_length
        self.prompt = args.prompt

        self.num_workers = getattr(args, "num_workers", 4)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

      
        self.pad_on_left = not self.is_train  # 训练 False(右)，评测 Tru
        self.tokenizer.padding_side = "left" if self.pad_on_left else "right"

        self.texts = []
        self.label_list = []
        self.get_sentences(data_path)

        self.samples = []
        self._pretokenize()

        if not self.is_train:
            self.samples.sort(key=lambda x: len(x["input_ids"]))

    def get_sentences(self, dir_path):
        with open(dir_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.texts = [item['sentence'] for item in data]
        self.label_list = [item['entities'] for item in data]

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

    def _pretokenize(self):
        for text, entities in zip(self.texts, self.label_list):
            prompt_text = build_prompt_text(
                self.tokenizer,
                text,
                self.prompt,
            )
            prompt_ids = self._tokenize(prompt_text)

            if self.is_train:
                target_text = build_target(entities)

                full_text = build_sft_text(
                    self.tokenizer,
                    text,
                    self.prompt,
                    target_text,
                )

                input_ids = self._tokenize(full_text)
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
                    input_ids.append(eos_id)

                input_ids = input_ids[:self.max_length]

                prompt_len = min(len(prompt_ids), len(input_ids))
                labels = ([-100] * prompt_len + input_ids[prompt_len:])
            else:
                input_ids = prompt_ids[:self.max_length]
                labels = [-100] * len(input_ids)

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "text": text,
                "entities": entities,
            })

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        input_ids_list = [x["input_ids"] for x in batch]
        labels_list = [x["labels"] for x in batch]

        text_list = [x["text"] for x in batch]
        entities_list = [x["entities"] for x in batch]

        batch_max_len = max(len(x) for x in input_ids_list)

        pad_id = self.tokenizer.pad_token_id
        padded_input_ids = []
        padded_labels = []
        padded_attn = []

        for inp_ids, lab in zip(input_ids_list, labels_list):
            inp_ids = inp_ids[:batch_max_len]
            lab = lab[:batch_max_len]
            pad_len = batch_max_len - len(inp_ids)

            if self.pad_on_left:
                padded_input_ids.append(
                    torch.tensor([pad_id] * pad_len + inp_ids)
                )
                padded_labels.append(
                    torch.tensor([-100] * pad_len + lab)
                )
                padded_attn.append(
                    torch.tensor([0] * pad_len + [1] * len(inp_ids))
                )
            else:
                padded_input_ids.append(
                    torch.tensor(inp_ids + [pad_id] * pad_len)
                )
                padded_labels.append(
                    torch.tensor(lab + [-100] * pad_len)
                )
                padded_attn.append(
                    torch.tensor([1] * len(inp_ids) + [0] * pad_len)
                )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attn),
            "labels": torch.stack(padded_labels),
            "text": text_list,
            "entities": entities_list,
        }

    def get_data_loader(self, batch_size, shuffle):
        kwargs = {
            "dataset": self,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }

        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2

        return DataLoader(**kwargs)