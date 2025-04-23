import yaml
from flag_scale.flagscale.agent.communication import Communicator


def convert_yaml_to_json(yaml_path: str):
    """
        Read the YAML file and return the data as a Dictionary.

    Args:
        robot_profile_path (str): Path to the robot profile YAML file.

    Returns:
        Dict: Data from the YAML files.

    """
    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    if "robot_tool" in yaml_data and isinstance(yaml_data["robot_tool"], dict):
        yaml_data["robot_tool"] = [
            {"tool_name": name, "class": f"tools.robotic_tools.{cfg['class']}"}
            for name, cfg in yaml_data["robot_tool"].items()
        ]
    return yaml_data


class Config:
    @classmethod
    def load_config(cls, config_path="config.yaml"):
        """Initialize configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config


config = Config.load_config()

communicator = Communicator(
    host=config["communicator"]["HOST"],
    port=config["communicator"]["PORT"],
    db=config["communicator"]["DB"],
    clear=config["communicator"]["CLEAR"],
    password=config["communicator"]["PASSWORD"],
)

if __name__ == "__main__":
    convert_yaml_to_json("robo/config/robot_profile.yaml")
