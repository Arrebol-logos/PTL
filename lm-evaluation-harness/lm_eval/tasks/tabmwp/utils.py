from datasets import Dataset

def process_docs(dataset: Dataset) -> Dataset:
    def _clean(example):
        # 如果没有 choices，就返回空串
        choices = example.get("choices", "")
        # 有时候 choices 可能是列表，拼接成字符串
        if isinstance(choices, list):
            choices = " ".join(choices)
        # 任何非字符串（比如 None）都转成空串
        if not isinstance(choices, str):
            choices = ""
            
        # table_title同理
        table_title = example.get("table_title", "")
        if isinstance(table_title, list):
            table_title = " ".join(table_title)
        if not isinstance(table_title, str):
            table_title = ""

        return {
            "question": example["question"].strip(),
            "table": table_title.strip() + example["table"].strip(),
            "choices": choices.strip(),
            "solution": example["solution"].strip(),
            "target": example["answer"].strip()
        }

    return dataset.map(_clean)
