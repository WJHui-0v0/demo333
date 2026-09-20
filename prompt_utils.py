def build_messages(sentence: str, system_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]


def build_prompt_text(tokenizer, sentence: str, system_prompt: str):
    messages = build_messages(sentence, system_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_sft_text(tokenizer, sentence: str, system_prompt: str, target: str):
    messages = build_messages(sentence, system_prompt)
    messages.append({"role": "assistant", "content": target})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_target(entities):
    if not entities:
        return "无实体"
    return "\n".join(
        f"{entity['name']}:GENE"
        for entity in entities
        if entity.get("name")
    ) or "无实体"


def get_generation_eos_ids(tokenizer):
    eos_ids = []

    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    vocab = tokenizer.get_vocab()
    im_end_token = "<|im_end|>"

    if im_end_token in vocab:
        im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)

        if im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

    if len(eos_ids) == 1:
        return eos_ids[0]

    return eos_ids
