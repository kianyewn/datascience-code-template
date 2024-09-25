import time

import numpy as np
import pandas as pd
from loguru import logger

from .file_helper import FileHandler, PathParser


def sample_dataframe(df, n_samples=20000, random_state=111):
    df_sample = df.sample(n=n_samples, random_state=random_state)
    return df_sample


def sample_dataframe_from_file_and_save(filepath, n_samples=20000, random_state=111):
    filepath = str(filepath)  # in case we are using Pathlib
    extension = PathParser.get_extension_from_filepath(filepath)
    path_wo_extension = PathParser.get_path_without_extension(filepath)

    df = pd.read_parquet(filepath)
    df_sample = sample_dataframe(df, n_samples=n_samples, random_state=random_state)

    sampled_file_path = f"{path_wo_extension.resolve()}{extension}"
    logger.info(sampled_file_path)

    df_sample.to_parquet(sampled_file_path)
    logger.info(
        f"Data is sampled with '{n_samples}' records and saved to location: '{sampled_file_path}'"
    )
    return df_sample, sampled_file_path


if __name__ == "__main__":
    np.random.seed(42)
    filepath = "dataframe.parquet"
    df = pd.DataFrame(np.random.randn(1000, 10))
    df.to_parquet(filepath)
    sampled_df, saved_path = sample_dataframe_from_file_and_save(
        filepath=filepath, n_samples=100, random_state=22
    )
    time.sleep(2)
    FileHandler.remove_file(saved_path)
