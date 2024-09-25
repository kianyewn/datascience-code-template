from loguru import logger
from pathlib import Path
from file_helper import FileHelper

def sample_dataframe(df, n_samples=10000, random_state=111):
    df_sample = df.sample(n=n_samples, random_state=111)
    return df_sample

def sample_dataframe_from_file(filepath, n_samples=10000, random_state=111):
    filepath = str(filepath) # in case we are using poxis
    

    