# 文件位置：lm_eval/tasks/svamp/utils.py

from datasets import Dataset

def process_docs(dataset: Dataset) -> Dataset:
    """
    对 SVAMP 数据集进行基本清洗：
    """
    def _clean(example):
        return {
            "question": example["question_concat"].strip(),
            "target": example["Answer"].strip()
        }
    return dataset.map(_clean)
