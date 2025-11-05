from datasets import load_dataset, Dataset


def VenCord(
    data_files: str = "data/vencord/vencord.parquet",
) -> Dataset:
    """
    Load and format dataset for training.

    Args:
        data_files: Path to the parquet file(s) containing training data

    Returns:
        Formatted dataset with messages in chat format
    """
    dataset = load_dataset("parquet", data_files=data_files)

    formatted = dataset["train"].map(
        lambda x: {
            "messages": [
                {"role": "user", "content": x["prompt"]},
                {"role": "assistant", "content": x["completion"]},
            ]
        },
        remove_columns=dataset["train"].column_names,
    )

    return formatted
