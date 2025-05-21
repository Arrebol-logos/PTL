from datasets import Dataset

def process_docs(dataset: Dataset) -> Dataset:
    """
    对 AQuA 数据集进行基本清洗，将 options 列表拼成字符串再 strip
    """
    def _clean(example):
        # 如果 options 已经是列表，就先 join；否则直接当字符串处理
        opts = example["options"]
        if isinstance(opts, list):
            opts = " ".join(opts)      # 或者用 "\n".join(opts)
        return {
            "question": example["question"].strip(),
            "options": opts.strip(),
            "correct": example["correct"].strip()
        }

    return dataset.map(_clean)