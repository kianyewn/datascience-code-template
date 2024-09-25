import pathlib
from pathlib import Path
from typing import List


class PathParser:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def get_base_name_from_filepath(filepath):
        return Path(filepath).stem

    @staticmethod
    def get_extensions_from_filepath(filepath: str) -> List:
        return Path(filepath).suffixes

    @staticmethod
    def get_extension_from_filepath(filepath: str) -> str:
        return Path(filepath).suffix

    @staticmethod
    def get_directory_from_filepath(filepath: str) -> pathlib.PosixPath:
        return Path(filepath).parent


if __name__ == "__main__":
    filepath = "/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils/file_helper.py"
    pp = PathParser(filepath)
    assert PathParser.get_base_name_from_filepath(filepath) == "file_helper"
    assert pp.get_base_name_from_filepath(filepath) == "file_helper"
    assert pp.get_extension_from_filepath(filepath) == ".py"
    assert pp.get_extensions_from_filepath(filepath) == [
        ".py"
    ], pp.get_extensions_from_filepath(filepath)
    assert pp.get_directory_from_filepath(filepath) == Path(
        "/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils"
    ), pp.get_directory_from_filepath(filepath)
    # '/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils'
    # '/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils'
