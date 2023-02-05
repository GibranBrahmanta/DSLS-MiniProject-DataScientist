import yaml

class ConfigLoader():

    config_path = "./config.yaml"

    def __init__(self) -> None:
        self.data = self.open_config()

    def open_config(self) -> dict:
        with open(self.config_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data

conf = ConfigLoader()
config = conf.data