from pathlib import Path
import json

project_base_path = Path(__file__).parent.parent.parent.parent
path_config = Path(
    project_base_path, "paths.config.json"
)  # NOTE: You're going to have to ensure that this copies over to the Python distribution
f = path_config.open()
data = json.load(f)
STUB_SAVE_FOLDER_PATH = Path(Path.home(), *data["stubs"])
CONVERSION_SAVE_FOLDER_PATH = Path(Path.home(), *data["conversions"])
TUTORIAL_SAVE_FOLDER_PATH = Path(Path.home(), *data["tutorial"])

f.close()
