import json
import os

class configreader:
    def __init__(self, config_files):
        if isinstance(config_files, list):
            self.config_files = config_files
            self.config_data = self.load_all_config()
        elif isinstance(config_files, int):
            self.config_file = config_files
            self.config_data = self.load_all_files()
        else:
            self.config_file = config_files
            self.config_data = self.load_config()

    def load_config(self):
        """Load configuration from a single JSON file."""
        try:
            with open(self.config_file, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"Error: The file {self.config_file} was not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: The configuration file is not a valid JSON.")
            return {}

    def load_all_config(self):
        """Load configuration from multiple JSON files."""
        config_data = {}
        for config_file in self.config_files:
            try:
                with open(config_file, 'r') as file:
                    file_data = json.load(file)
                    config_data.update(file_data)
            except FileNotFoundError:
                print(f"Error: The file {config_file} was not found.")
            except json.JSONDecodeError:
                print(f"Error: The configuration file {config_file} is not a valid JSON.")
        return config_data

    def load_all_files(self):
        """Load all JSON files in the 'conf' folder except '.gitkeep'."""
        config_data = {}
        conf_folder = os.path.join(os.getcwd(), 'conf')  # Adjust the path if needed
        try:
            for filename in os.listdir(conf_folder):
                if filename.endswith('.json') and filename != '.gitkeep':
                    file_path = os.path.join(conf_folder, filename)
                    with open(file_path, 'r') as file:
                        file_data = json.load(file)
                        config_data.update(file_data)
        except FileNotFoundError:
            print(f"Error: The folder {conf_folder} was not found.")
        except json.JSONDecodeError as e:
            print(f"Error: A file in {conf_folder} is not a valid JSON. {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return config_data

    def get(self, key, default=None):
        """Get a configuration value by key with an optional default."""
        keys = key.split('.')
        value = self.config_data
        for k in keys:
            value = value.get(k, default)
            if value == default:
                break
        return value

    def get_all(self):
        """Return all configuration settings."""
        return self.config_data

# Global dictionary to store configuration data
configDict = {}

def initialize_config(config_files):
    global configDict
    config = configreader(config_files)
    configDict = config.get_all()


def return_config(config_files):
    global configDict
    config = configreader(config_files)
    configDict = config.get_all()
    return configDict