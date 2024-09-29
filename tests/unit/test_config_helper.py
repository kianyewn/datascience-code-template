import pytest
from src.utils.config_helper import ConfigYAML


@pytest.fixture
def mydict():
    mydict = {"a": 1, "b": 2, "c": 3}
    return mydict


def test_ConfigYAML(mocker, mydict):
    YAML_PATH = "data/test_config.yaml"
    m1 = mocker.patch("src.utils.config_helper.ConfigYAML.save", return_value=True)
    m2 = mocker.patch("src.utils.config_helper.ConfigYAML.delete", return_value=True)
    _ = ConfigYAML.save(mydict, YAML_PATH)
    ConfigYAML.delete(YAML_PATH)

    m1.assert_called_with(mydict, YAML_PATH)
    m2.assert_called_with(YAML_PATH)
