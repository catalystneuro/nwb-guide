"""Collection of utility functions used by the NeuroConv Flask API."""

import copy
import hashlib
import inspect
import json
import math
import os
import re
import traceback
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copytree, rmtree
from typing import Any, Dict, List, Optional, Union

from pynwb import NWBFile
from tqdm_publisher import TQDMProgressHandler

from .info import (
    CONVERSION_SAVE_FOLDER_PATH,
    GUIDE_ROOT_FOLDER,
    STUB_SAVE_FOLDER_PATH,
    is_packaged,
    resource_path,
)
from .info.sse import format_sse

progress_handler = TQDMProgressHandler()

EXCLUDED_RECORDING_INTERFACE_PROPERTIES = ["contact_vector", "contact_shapes", "group", "location"]

EXTRA_INTERFACE_PROPERTIES = {
    "brain_area": {
        "data_type": "str",
        "default": "unknown",
    }
}

EXTRA_RECORDING_INTERFACE_PROPERTIES = list(EXTRA_INTERFACE_PROPERTIES.keys())

RECORDING_INTERFACE_PROPERTY_OVERRIDES = {
    "brain_area": {
        "description": "The brain area where the electrode is located.",
        **EXTRA_INTERFACE_PROPERTIES["brain_area"],
    }
}

EXTRA_SORTING_INTERFACE_PROPERTIES = ["unit_id", *EXTRA_INTERFACE_PROPERTIES.keys()]

SORTING_INTERFACE_PROPERTIES_TO_RECAST = {
    "quality": {
        "data_type": "str",
    },
    "KSLabel": {
        "data_type": "str",
    },
    "KSLabel_repeat": {
        "data_type": "str",
    },
}

SORTING_INTERFACE_PROPERTY_OVERRIDES = {
    "unit_id": {"description": "The unique ID for this unit", "data_type": "str"},
    "brain_area": {
        "description": "The brain area where the unit is located.",
        **EXTRA_INTERFACE_PROPERTIES["brain_area"],
    },
    **SORTING_INTERFACE_PROPERTIES_TO_RECAST,
}

# NOTE: No need to show this if it isn't editable
del SORTING_INTERFACE_PROPERTY_OVERRIDES["brain_area"]
brain_area_idx = EXTRA_SORTING_INTERFACE_PROPERTIES.index("brain_area")
EXTRA_SORTING_INTERFACE_PROPERTIES.pop(brain_area_idx)


EXCLUDED_SORTING_INTERFACE_PROPERTIES = ["location", "spike_times", "electrodes"]  # Not validated

# NOTE: These are the only accepted dtypes...
DTYPE_DESCRIPTIONS = {
    "bool": "logical",
    "str": "string",
    "ndarray": "n-dimensional array",
    "float8": "8-bit number",
    "float16": "16-bit number",
    "float32": "32-bit number",
    "float64": "64-bit number",
    "int8": "8-bit integer",
    "int16": "16-bit integer",
    "int32": "32-bit integer",
    "int64": "64-bit integer",
}

DTYPE_SCHEMA = {
    "type": "string",
    # "strict": False,
    "enum": list(DTYPE_DESCRIPTIONS.keys()),
    "enumLabels": DTYPE_DESCRIPTIONS,
}


def is_path_contained(child, parent):
    parent = Path(parent)
    child = Path(child)

    # Attempt to construct a relative path from parent to child
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def replace_nan_with_none(data):
    if isinstance(data, dict):
        # If it's a dictionary, iterate over its items and replace NaN values with None
        return {key: replace_nan_with_none(value) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, iterate over its elements and replace NaN values with None
        return [replace_nan_with_none(item) for item in data]
    elif isinstance(data, (float, int)) and (data != data):
        return None  # Replace NaN with None
    else:
        return data


def resolve_references(schema: dict, root_schema: Optional[dict] = None) -> dict:
    """
    Recursively resolve references in a JSON schema based on the root schema.

    Args:
        schema (dict): The JSON schema to resolve.
        root_schema (dict): The root JSON schema.

    Returns:
        dict: The resolved JSON schema.
    """
    from jsonschema import RefResolver

    if root_schema is None:
        root_schema = schema

    if "$ref" in schema:
        resolver = RefResolver.from_schema(root_schema)
        return resolver.resolve(schema["$ref"])[1]

    if "properties" in schema:
        for key, prop_schema in schema["properties"].items():
            schema["properties"][key] = resolve_references(prop_schema, root_schema)

    if "items" in schema:
        schema["items"] = resolve_references(schema["items"], root_schema)

    return schema


def replace_none_with_nan(json_object: dict, json_schema: dict) -> dict:
    """
    Recursively search a JSON object and replace None values with NaN where appropriate.

    Args:
        json_object (dict): The JSON object to search and modify.
        json_schema (dict): The JSON schema to validate against.

    Returns:
        dict: The modified JSON object with None values replaced by NaN.
    """

    def coerce_schema_compliance_recursive(obj, schema):
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Coerce on pattern properties as well
                pattern_properties = schema.get("patternProperties")
                if pattern_properties:
                    for pattern, pattern_schema in pattern_properties.items():
                        regex = re.compile(pattern)
                        if regex.match(key):
                            coerce_schema_compliance_recursive(value, pattern_schema)

                elif key in schema.get("properties", {}):
                    prop_schema = schema["properties"][key]
                    if prop_schema.get("type") == "number" and (value is None or value == "NaN"):
                        obj[key] = (
                            math.nan
                        )  # Turn None into NaN if a number is expected (JavaScript JSON.stringify turns NaN into None)
                    elif prop_schema.get("type") == "number" and isinstance(value, int):
                        obj[key] = float(
                            value
                        )  # Turn integer into float if a number, the JSON Schema equivalent to float, is expected (JavaScript coerces floats with trailing zeros to integers)
                    else:
                        coerce_schema_compliance_recursive(value, prop_schema)
        elif isinstance(obj, list):
            for item in obj:
                coerce_schema_compliance_recursive(
                    item, schema.get("items", schema if "properties" else {})
                )  # NEUROCONV PATCH

        return obj

    return coerce_schema_compliance_recursive(
        copy.deepcopy(json_object), resolve_references(copy.deepcopy(json_schema))
    )


def autocomplete_format_string(info: dict) -> str:
    from neuroconv.tools.path_expansion import construct_path_template
    from neuroconv.utils.json_schema import NWBMetaDataEncoder

    base_directory = info["base_directory"]
    filesystem_entry_path = info["path"]

    if not is_path_contained(filesystem_entry_path, base_directory):
        raise ValueError("Path is not contained in the provided base directory.")

    full_format_string = construct_path_template(
        filesystem_entry_path,
        subject_id=info["subject_id"],
        session_id=info["session_id"],
        **info["additional_metadata"],
    )

    parent = Path(base_directory).resolve()
    child = Path(full_format_string).resolve()

    format_string = str(child.relative_to(parent))

    to_locate_info = dict(base_directory=base_directory)

    if Path(filesystem_entry_path).is_dir():
        to_locate_info["folder_path"] = format_string
    else:
        to_locate_info["file_path"] = format_string

    all_matched = locate_data(dict(autocomplete=to_locate_info))

    return json.loads(json.dumps(obj=dict(matched=all_matched, format_string=format_string), cls=NWBMetaDataEncoder))


def locate_data(info: dict) -> dict:
    """Locate data from the specifies directories using fstrings."""
    from neuroconv.tools import LocalPathExpander
    from neuroconv.utils.json_schema import NWBMetaDataEncoder

    expander = LocalPathExpander()

    # Transform the input into a list of dictionaries
    for value in info.values():
        if (value.get("base_directory") is not None) and (value.get("base_directory") != ""):
            value["base_directory"] = Path(value["base_directory"])

    out = expander.expand_paths(info)

    # Organize results by subject, session, and data type
    organized_output = {}
    for item in out:
        subject_id = item["metadata"]["Subject"]["subject_id"]
        session_id = item["metadata"]["NWBFile"]["session_id"]
        if subject_id not in organized_output:
            organized_output[subject_id] = {}

        if session_id not in organized_output[subject_id]:
            organized_output[subject_id][session_id] = {}

        organized_output[subject_id][session_id] = item

    return json.loads(json.dumps(obj=organized_output, cls=NWBMetaDataEncoder))


def module_to_dict(my_module) -> dict:
    # Create an empty dictionary
    module_dict = {}

    # Iterate through the module's attributes
    for attr_name in dir(my_module):
        if not attr_name.startswith("__"):  # Exclude special attributes
            attr_value = getattr(my_module, attr_name)
            module_dict[attr_name] = attr_value

    return module_dict


doc_pattern = r":py:class:`\~.+\..+\.(\w+)`"
remove_extra_spaces_pattern = r"\s+"


def get_class_ref_in_docstring(input_string):
    match = re.search(doc_pattern, input_string)

    if match:
        return match.group(1)


def derive_interface_info(interface) -> dict:

    info = {"keywords": getattr(interface, "keywords", []), "description": ""}

    if hasattr(interface, "associated_suffixes"):
        info["suffixes"] = interface.associated_suffixes

    if hasattr(interface, "info"):
        info["description"] = interface.info

    elif interface.__doc__:
        info["description"] = re.sub(
            remove_extra_spaces_pattern, " ", re.sub(doc_pattern, r"<code>\1</code>", interface.__doc__)
        )

    info["name"] = interface.__name__

    return info


def get_all_converter_info() -> dict:
    from neuroconv.converters import converter_list

    return {
        getattr(converter, "display_name", converter.__name__) or converter.__name__: derive_interface_info(converter)
        for converter in converter_list
    }


def get_all_interface_info() -> dict:
    """Format an information structure to be used for selecting interfaces based on modality and technique."""
    from neuroconv.datainterfaces import interface_list

    exclude_interfaces_from_selection = [
        # Deprecated
        "SpikeGLXLFPInterface",
        # Aliased
        "CEDRecordingInterface",
        "OpenEphysBinaryRecordingInterface",
        "OpenEphysLegacyRecordingInterface",
        # Ignored
        "AxonaPositionDataInterface",
        "AxonaUnitRecordingInterface",
        "CsvTimeIntervalsInterface",
        "ExcelTimeIntervalsInterface",
        "Hdf5ImagingInterface",
        "MaxOneRecordingInterface",
        "OpenEphysSortingInterface",
        "SimaSegmentationInterface",
    ]

    return {
        getattr(interface, "display_name", interface.__name__) or interface.__name__: derive_interface_info(interface)
        for interface in interface_list
        if not interface.__name__ in exclude_interfaces_from_selection
    }


# Combine Multiple Interfaces
def get_custom_converter(interface_class_dict: dict, alignment_info: Union[dict, None] = None) -> "NWBConverter":
    from neuroconv import NWBConverter, converters, datainterfaces

    alignment_info = alignment_info or dict()

    class CustomNWBConverter(NWBConverter):
        data_interface_classes = {
            custom_name: getattr(datainterfaces, interface_name, getattr(converters, interface_name, None))
            for custom_name, interface_name in interface_class_dict.items()
        }

        # Handle temporal alignment inside the converter
        # TODO: this currently works off of cross-scoping injection of `alignment_info` - refactor to be more explicit
        def temporally_align_data_interfaces(self):
            set_interface_alignment(self, alignment_info=alignment_info)

        # From previous issue regarding SpikeGLX not generating previews of correct size
        def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None) -> None:
            conversion_options = conversion_options or dict()
            for interface_key, data_interface in self.data_interface_objects.items():
                if isinstance(data_interface, NWBConverter):
                    subconverter_kwargs = dict(nwbfile=nwbfile, metadata=metadata)

                    # Certain subconverters fully expose control over their interfaces conversion options
                    # (such as iterator options, including progress bar details)
                    subconverter_keyword_arguments = list(
                        inspect.signature(data_interface.add_to_nwbfile).parameters.keys()
                    )
                    if "conversion_options" in subconverter_keyword_arguments:
                        subconverter_kwargs["conversion_options"] = conversion_options.get(interface_key, None)
                    # Others do not, and instead expose simplified global keywords similar to a classic interface
                    else:
                        subconverter_kwargs.update(conversion_options.get(interface_key, dict()))

                    data_interface.add_to_nwbfile(**subconverter_kwargs)
                else:
                    data_interface.add_to_nwbfile(
                        nwbfile=nwbfile, metadata=metadata, **conversion_options.get(interface_key, dict())
                    )

    return CustomNWBConverter


def instantiate_custom_converter(
    source_data: Dict, interface_class_dict: Dict, alignment_info: Union[Dict, None] = None
) -> "NWBConverter":
    alignment_info = alignment_info or dict()

    CustomNWBConverter = get_custom_converter(interface_class_dict=interface_class_dict, alignment_info=alignment_info)

    return CustomNWBConverter(source_data=source_data)


def get_source_schema(interface_class_dict: dict) -> dict:
    """Function used to get schema from a CustomNWBConverter that can handle multiple interfaces."""
    CustomNWBConverter = get_custom_converter(interface_class_dict)
    return CustomNWBConverter.get_source_schema()


def map_interfaces(callback, converter, to_match: Union["BaseDataInterface", None] = None, parent_name=None) -> list:
    from neuroconv import NWBConverter

    output = []

    for name, interface in converter.data_interface_objects.items():

        associated_name = f"{parent_name} — {name}" if parent_name else name
        if isinstance(interface, NWBConverter):
            result = map_interfaces(
                callback=callback, converter=interface, to_match=to_match, parent_name=associated_name
            )
            output.extend(result)
        elif to_match is None or isinstance(interface, to_match):
            result = callback(associated_name, interface)
            output.append(result)

    return output


def get_metadata_schema(source_data: Dict[str, dict], interfaces: dict) -> Dict[str, dict]:
    """Function used to fetch the metadata schema from a CustomNWBConverter instantiated from the source_data."""
    from neuroconv.utils import NWBMetaDataEncoder

    resolved_source_data = replace_none_with_nan(
        source_data, resolve_references(get_custom_converter(interfaces).get_source_schema())
    )

    converter = instantiate_custom_converter(resolved_source_data, interfaces)
    schema = converter.get_metadata_schema()
    metadata = converter.get_metadata()

    # Clear the Electrodes information for being set as a collection of Interfaces
    has_ecephys = "Ecephys" in metadata
    has_units = False
    ecephys_metadata = metadata.get("Ecephys")
    ecephys_schema = schema["properties"].get("Ecephys", {"properties": {}})

    if not ecephys_schema.get("required"):
        ecephys_schema["required"] = []

    ecephys_properties = ecephys_schema["properties"]
    original_electrodes_schema = ecephys_properties.get("Electrodes")

    resolved_electrodes = {}
    resolved_units = {}
    resolved_electrodes_schema = {"type": "object", "properties": {}, "required": []}
    resolved_units_schema = {"type": "object", "properties": {}, "required": []}

    def on_sorting_interface(name, sorting_interface):

        unit_columns = get_unit_columns_json(sorting_interface)

        # Aggregate unit column information across sorting interfaces
        existing_unit_columns = metadata["Ecephys"].get("UnitColumns")
        if existing_unit_columns:
            for entry in unit_columns:
                if any(obj["name"] == entry["name"] for obj in existing_unit_columns):
                    continue
                else:
                    existing_unit_columns.append(entry)
        else:
            metadata["Ecephys"]["UnitColumns"] = unit_columns

        units_data = resolved_units[name] = get_unit_table_json(sorting_interface)

        n_units = len(units_data)

        resolved_units_schema["properties"][name] = {
            "type": "array",
            "minItems": n_units,
            "maxItems": n_units,
            "items": {
                "allOf": [
                    {"$ref": "#/properties/Ecephys/definitions/Unit"},
                    {"required": list(map(lambda info: info["name"], unit_columns))},
                ]
            },
        }

        resolved_units_schema["required"].append(name)

        return sorting_interface

    def on_recording_interface(name, recording_interface):
        electrode_columns = get_electrode_columns_json(recording_interface)

        # Aggregate electrode column information across recording interfaces
        existing_electrode_columns = ecephys_metadata.get("ElectrodeColumns")
        if existing_electrode_columns:
            for entry in electrode_columns:
                if any(obj["name"] == entry["name"] for obj in existing_electrode_columns):
                    continue
                else:
                    existing_electrode_columns.append(entry)
        else:
            ecephys_metadata["ElectrodeColumns"] = electrode_columns

        electrode_data = resolved_electrodes[name] = get_electrode_table_json(recording_interface)

        n_electrodes = len(electrode_data)

        resolved_electrodes_schema["properties"][name] = {
            "type": "array",
            "minItems": n_electrodes,
            "maxItems": n_electrodes,
            "items": {
                "allOf": [
                    {"$ref": "#/properties/Ecephys/definitions/Electrode"},
                    {"required": list(map(lambda info: info["name"], electrode_columns))},
                ]
            },
        }

        resolved_electrodes_schema["required"].append(name)

        return recording_interface

    from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import (
        BaseRecordingExtractorInterface,
    )
    from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import (
        BaseSortingExtractorInterface,
    )

    # Map recording interfaces to metadata
    map_interfaces(on_recording_interface, converter=converter, to_match=BaseRecordingExtractorInterface)

    # Map sorting interfaces to metadata
    map_interfaces(on_sorting_interface, converter=converter, to_match=BaseSortingExtractorInterface)

    if has_ecephys:

        if "definitions" not in ecephys_schema:
            ecephys_schema["definitions"] = ecephys_properties["definitions"]

        has_electrodes = "ElectrodeColumns" in ecephys_metadata

        original_units_schema = ecephys_properties.pop("UnitProperties", None)
        ecephys_metadata.pop("UnitProperties", None)  # Always remove top-level UnitProperties from metadata
        has_units = original_units_schema is not None

        # Populate Electrodes metadata
        if has_electrodes:

            # Add Electrodes to the schema
            ecephys_metadata["Electrodes"] = resolved_electrodes
            ecephys_schema["required"].append("Electrodes")

            ecephys_properties["ElectrodeColumns"] = {
                "type": "array",
                "minItems": 0,
                "items": {"$ref": "#/properties/Ecephys/definitions/ElectrodeColumn"},
            }

            ecephys_schema["required"].append("ElectrodeColumns")

            ecephys_properties["Electrodes"] = resolved_electrodes_schema

        else:
            ecephys_properties.pop("Electrodes", None)

        # Populate Units metadata
        if has_units:

            ecephys_properties["UnitColumns"] = {
                "type": "array",
                "minItems": 0,
                "items": {"$ref": "#/properties/Ecephys/definitions/UnitColumn"},
            }

            schema["properties"]["Ecephys"]["required"].append("UnitColumns")

            ecephys_properties["Units"] = resolved_units_schema
            ecephys_metadata["Units"] = resolved_units
            schema["properties"]["Ecephys"]["required"].append("Units")

        # Delete Ecephys metadata if no interfaces processed
        defs = ecephys_schema["definitions"]

        electrode_def = defs["Electrodes"]

        # NOTE: Update to output from NeuroConv
        electrode_def["properties"]["data_type"] = DTYPE_SCHEMA

        # Configure electrode columns
        defs["ElectrodeColumn"] = electrode_def
        defs["ElectrodeColumn"]["required"] = list(electrode_def["properties"].keys())

        new_electrodes_properties = {
            properties["name"]: {key: value for key, value in properties.items() if key != "name"}
            for properties in original_electrodes_schema.get("default", {})
            if properties["name"] not in EXCLUDED_RECORDING_INTERFACE_PROPERTIES
        }

        defs["Electrode"] = {
            "type": "object",
            "properties": new_electrodes_properties,
            "additionalProperties": True,  # Allow for new columns
        }

        if has_units:

            unitprops_def = defs["UnitProperties"]

            # NOTE: Update to output from NeuroConv
            unitprops_def["properties"]["data_type"] = DTYPE_SCHEMA

            # Configure electrode columns
            defs["UnitColumn"] = unitprops_def
            defs["UnitColumn"]["required"] = list(unitprops_def["properties"].keys())

            new_units_properties = {
                properties["name"]: {key: value for key, value in properties.items() if key != "name"}
                for properties in original_units_schema.get("default", {})
                if properties["name"] not in EXCLUDED_SORTING_INTERFACE_PROPERTIES
            }

            defs["Unit"] = {
                "type": "object",
                "properties": new_units_properties,
                "additionalProperties": True,  # Allow for new columns
            }

    # TODO: generalize logging stuff
    log_base = GUIDE_ROOT_FOLDER / "logs"
    log_base.mkdir(exist_ok=True)
    with open(file=log_base / "file_metadata_page_schema.json", mode="w") as fp:
        json.dump(obj=dict(schema=schema), fp=fp, cls=NWBMetaDataEncoder, indent=2)
    with open(file=log_base / "file_metadata_page_results.json", mode="w") as fp:
        json.dump(obj=dict(results=metadata), fp=fp, cls=NWBMetaDataEncoder, indent=2)

    return json.loads(
        json.dumps(obj=replace_nan_with_none(dict(results=metadata, schema=schema)), cls=NWBMetaDataEncoder)
    )


def get_check_function(check_function_name: str) -> callable:
    """Function used to fetch an arbitrary NWB Inspector function."""
    from nwbinspector import configure_checks, load_config

    dandi_check_list = configure_checks(config=load_config(filepath_or_keyword="dandi"))
    dandi_check_registry = {check.__name__: check for check in dandi_check_list}

    check_function: callable = dandi_check_registry.get(check_function_name)
    if check_function is None:
        raise ValueError(f"Function {check_function_name} not found in nwbinspector")

    return check_function


def run_check_function(check_function: callable, arg: dict) -> dict:
    """.Function used to run an arbitrary NWB Inspector function."""
    from nwbinspector import Importance, InspectorMessage

    output = check_function(arg)
    if isinstance(output, InspectorMessage):
        if output.importance != Importance.ERROR:
            output.importance = check_function.importance
    elif output is not None:
        for x in output:
            x.importance = check_function.importance

    return output


def validate_subject_metadata(
    subject_metadata: dict, check_function_name: str, timezone: Optional[str] = None
):  # -> Union[None, InspectorMessage, List[InspectorMessage]]:
    """Function used to validate subject metadata."""
    import pytz
    from pynwb.file import Subject

    check_function = get_check_function(check_function_name)

    if isinstance(subject_metadata.get("date_of_birth"), str):
        subject_metadata["date_of_birth"] = datetime.fromisoformat(subject_metadata["date_of_birth"])
        if timezone is not None:
            subject_metadata["date_of_birth"] = subject_metadata["date_of_birth"].replace(
                tzinfo=zoneinfo.ZoneInfo(timezone)
            )

    return run_check_function(check_function, Subject(**subject_metadata))


def validate_nwbfile_metadata(
    nwbfile_metadata: dict, check_function_name: str, timezone: Optional[str] = None
):  # -> Union[None, InspectorMessage, List[InspectorMessage]]:
    """Function used to validate NWBFile metadata."""
    from pynwb.testing.mock.file import mock_NWBFile

    check_function = get_check_function(check_function_name)

    if isinstance(nwbfile_metadata.get("session_start_time"), str):
        nwbfile_metadata["session_start_time"] = datetime.fromisoformat(nwbfile_metadata["session_start_time"])
        if timezone is not None:
            nwbfile_metadata["session_start_time"] = nwbfile_metadata["session_start_time"].replace(
                tzinfo=zoneinfo.ZoneInfo(timezone)
            )

    return run_check_function(check_function, mock_NWBFile(**nwbfile_metadata))


def validate_metadata(
    metadata: dict,
    check_function_name: str,
    timezone: Optional[str] = None,
) -> dict:
    """Function used to validate data using an arbitrary NWB Inspector function."""
    from nwbinspector import InspectorOutputJSONEncoder
    from pynwb.file import NWBFile, Subject

    check_function = get_check_function(check_function_name)

    if issubclass(check_function.neurodata_type, Subject):
        result = validate_subject_metadata(metadata, check_function_name, timezone)
    elif issubclass(check_function.neurodata_type, NWBFile):
        result = validate_nwbfile_metadata(metadata, check_function_name, timezone)
    else:
        raise ValueError(
            f"Function {check_function_name} with neurodata_type {check_function.neurodata_type} "
            "is not supported by this function!"
        )

    return json.loads(json.dumps(result, cls=InspectorOutputJSONEncoder))


def set_interface_alignment(converter: dict, alignment_info: dict) -> dict:

    import numpy as np
    from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import (
        BaseSortingExtractorInterface,
    )
    from neuroconv.tools.testing.mock_interfaces import MockRecordingInterface

    errors = {}

    for name, interface in converter.data_interface_objects.items():

        interface = converter.data_interface_objects[name]
        info = alignment_info.get(name, {})

        # Set alignment
        method = info.get("selected", None)
        if method is None:
            continue

        value = info["values"].get(method, None)
        if value is None:
            continue

        try:
            if method == "timestamps":

                # Open the input file for reading
                # Can be .txt, .csv, .tsv, etc.
                # But timestamps must be scalars separated by newline characters
                with open(file=value, mode="r") as io:
                    aligned_timestamps = np.array([float(line.strip()) for line in io.readlines()])

                # Special case for sorting interfaces; to set timestamps they must have a recording registered
                must_set_mock_recording = (
                    isinstance(interface, BaseSortingExtractorInterface)
                    and not interface.sorting_extractor.has_recording()
                )
                if must_set_mock_recording is True:
                    sorting_extractor = interface.sorting_extractor
                    sampling_frequency = sorting_extractor.get_sampling_frequency()
                    end_frame = timestamps_array.shape[0]
                    mock_recording_interface = MockRecordingInterface(
                        sampling_frequency=sampling_frequency,
                        durations=[end_frame / sampling_frequency],
                        num_channels=1,
                    )
                    interface.register_recording(recording_interface=mock_recording_interface)

                interface.set_aligned_timestamps(aligned_timestamps=aligned_timestamps)

            # Special case for sorting interfaces; a recording interface to be converted may be registered/linked
            elif method == "linked":
                interface.register_recording(converter.data_interface_objects[value])

            elif method == "start":
                interface.set_aligned_starting_time(aligned_starting_time=value)

        except Exception as e:
            errors[name] = str(e)

    return errors


def get_compatible_interfaces(info: dict) -> dict:

    from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import (
        BaseRecordingExtractorInterface,
    )
    from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import (
        BaseSortingExtractorInterface,
    )

    converter = instantiate_custom_converter(source_data=info["source_data"], interface_class_dict=info["interfaces"])

    compatible = {}

    for name, interface in converter.data_interface_objects.items():

        is_sorting = isinstance(interface, BaseSortingExtractorInterface)

        if is_sorting is True:
            compatible[name] = []

        # If at least one recording and sorting interface is selected on the formats page
        # Then it is possible the two could be linked (the sorting was applied to the recording)
        # But there are very strict conditions from SpikeInterface determining compatibility
        # Those conditions are not easily exposed so we just 'try' to register them and skip on error
        sibling_recording_interfaces = {
            interface_key: interface
            for interface_key, interface in converter.data_interface_objects.items()
            if isinstance(interface, BaseRecordingExtractorInterface)
        }

        for recording_interface_key, recording_interface in sibling_recording_interfaces.items():
            try:
                interface.register_recording(recording_interface=recording_interface)
                compatible[name].append(recording_interface_key)
            except Exception:
                pass

    return compatible


def get_interface_alignment(info: dict) -> dict:

    from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
    from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import (
        BaseSortingExtractorInterface,
    )

    alignment_info = info.get("alignment", dict())

    converter = instantiate_custom_converter(source_data=info["source_data"], interface_class_dict=info["interfaces"])

    compatibility = get_compatible_interfaces(info)

    errors = set_interface_alignment(converter=converter, alignment_info=alignment_info)

    metadata = dict()
    timestamps = dict()

    for name, interface in converter.data_interface_objects.items():

        metadata[name] = dict()

        is_sorting = isinstance(interface, BaseSortingExtractorInterface)
        metadata[name]["sorting"] = is_sorting

        if is_sorting is True:
            metadata[name]["compatible"] = compatibility.get(name, None)

        if not isinstance(interface, BaseTemporalAlignmentInterface):
            timestamps[name] = []
            continue

        # Note: it is technically possible to have a BaseTemporalAlignmentInterface that has not yet implemented
        # the `get_timestamps` method; try to get this but skip on error
        try:
            interface_timestamps = interface.get_timestamps()
            if len(interface_timestamps) == 1:
                interface_timestamps = interface_timestamps[0]

            # Some interfaces, such as video or audio, may return a list of arrays
            # corresponding to each file of their `file_paths` input
            # Note: GUIDE only currently supports single files for these interfaces
            # Thus, unpack only the first array
            if isinstance(interface_timestamps, list):
                interface_timestamps = interface_timestamps[0]

            timestamps[name] = interface_timestamps.tolist()
        except Exception:
            timestamps[name] = []

    return dict(
        metadata=metadata,
        timestamps=timestamps,
        errors=errors,
    )


def create_file(
    info: dict,
    log_url: Optional[str] = None,
) -> dict:
    import neuroconv
    import requests
    from tqdm_publisher import TQDMProgressSubscriber

    project_name = info.get("project_name")

    run_stub_test = info.get("stub_test", False)

    overwrite = info.get("overwrite", False)

    # Progress update info
    url = info.get("url")
    request_id = info.get("request_id")

    # Backend configuration info
    backend_configuration = info.get("configuration", {})
    backend = backend_configuration.get("backend", "hdf5")

    converter, metadata, path_info = get_conversion_info(info)

    nwbfile_path = path_info["file"]

    try:

        # Delete files manually if using Zarr
        if overwrite:
            if nwbfile_path.exists():
                if nwbfile_path.is_dir():
                    rmtree(nwbfile_path)
                else:
                    nwbfile_path.unlink()

        def update_conversion_progress(message):
            update_dict = dict(request_id=request_id, **message)
            if url or not run_stub_test:
                requests.post(url=url, json=update_dict)
            else:
                progress_handler.announce(update_dict)

        progress_bar_options = dict(
            mininterval=0,
            on_progress_update=update_conversion_progress,
        )

        # Assume all interfaces have the same conversion options for now
        conversion_options_schema = converter.get_conversion_options_schema()
        conversion_options = {interface: dict() for interface in info["source_data"]}

        for interface_or_subconverter in conversion_options:
            conversion_options_schema_per_interface_or_converter = conversion_options_schema.get(
                "properties", dict()
            ).get(interface_or_subconverter, dict())

            # Object is a nested converter
            if conversion_options_schema_per_interface_or_converter.get("title", "") == "Conversion options schema":
                subconverter = interface_or_subconverter

                conversion_options_schema_per_subinterface = conversion_options_schema_per_interface_or_converter.get(
                    "properties", dict()
                )

                for subinterface, subschema in conversion_options_schema_per_subinterface.items():
                    conversion_options[subconverter][subinterface] = dict()
                    options_to_update = conversion_options[subconverter][subinterface]

                    properties_per_subinterface = subschema.get("properties", dict())

                    if run_stub_test is True and "stub_test" in properties_per_subinterface:
                        options_to_update["stub_test"] = True

                    # Only display per-file progress updates if not running a preview
                    if run_stub_test is False and "iterator_opts" in properties_per_subinterface:
                        options_to_update["iterator_opts"] = dict(
                            display_progress=True,
                            progress_bar_class=TQDMProgressSubscriber,
                            progress_bar_options=progress_bar_options,
                        )

            # Object is a standard interface
            else:
                interface = interface_or_subconverter

                conversion_options_schema_per_interface = conversion_options_schema.get("properties", dict())
                options_to_update = conversion_options[interface]

                if run_stub_test is True and "stub_test" in conversion_options_schema_per_interface:
                    options_to_update[interface]["stub_test"] = True

                # Only display per-file progress updates if not running a preview
                if run_stub_test is False and "iterator_opts" in conversion_options_schema_per_interface:
                    options_to_update[interface]["iterator_opts"] = dict(
                        display_progress=True,
                        progress_bar_class=TQDMProgressSubscriber,
                        progress_bar_options=progress_bar_options,
                    )

        # Add GUIDE watermark
        package_json_file_path = resource_path("package.json" if is_packaged() else "../package.json")
        with open(file=package_json_file_path) as fp:
            package_json = json.load(fp=fp)
        app_version = package_json["version"]
        metadata["NWBFile"]["source_script"] = f"Created using NWB GUIDE v{app_version}"
        metadata["NWBFile"]["source_script_file_name"] = neuroconv.__file__  # Must be included to be valid

        run_conversion_kwargs = dict(
            metadata=metadata,
            nwbfile_path=nwbfile_path,
            overwrite=overwrite,
            conversion_options=conversion_options,
            backend=backend,
        )

        # Only set full backend configuration if running a full conversion
        if run_stub_test is False:
            run_conversion_kwargs.update(dict(backend_configuration=update_backend_configuration(info)))

        converter.run_conversion(**run_conversion_kwargs)

    except Exception as e:
        if log_url:
            requests.post(
                url=log_url,
                json=dict(
                    header=f"Conversion failed for {project_name} — {nwbfile_path} (convert_to_nwb)",
                    inputs=dict(info=info),
                    traceback=traceback.format_exc(),
                    type="error",
                ),
            )

        raise e


def update_backend_configuration(info: dict) -> dict:

    from neuroconv.tools.nwb_helpers import (
        get_default_backend_configuration,
        make_nwbfile_from_metadata,
    )

    PROPS_TO_IGNORE = ["full_shape"]

    info_from_frontend = info.get("configuration", {})
    backend = info_from_frontend.get("backend", "hdf5")
    backend_configuration_from_frontend = info_from_frontend.get("results", {}).get(backend, {})

    converter, metadata, __ = get_conversion_info(info)

    nwbfile = make_nwbfile_from_metadata(metadata=metadata)
    converter.add_to_nwbfile(nwbfile, metadata=metadata)

    backend_configuration = get_default_backend_configuration(nwbfile=nwbfile, backend=backend)

    for location_in_file, dataset_configuration in backend_configuration_from_frontend.items():
        for key, value in dataset_configuration.items():
            if key not in PROPS_TO_IGNORE:
                # Pydantic models only allow setting of attributes
                setattr(backend_configuration.dataset_configurations[location_in_file], key, value)

    return backend_configuration


def get_backend_configuration(info: dict) -> dict:

    import numpy as np

    PROPS_TO_REMOVE = [
        # Immutable
        "object_id",
        "dataset_name",
        "location_in_file",
        "dtype",
    ]

    info["overwrite"] = True  # Always overwrite the file

    backend = info.get("backend", "hdf5")
    configuration = update_backend_configuration(info)

    def custom_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Provide metadata on configuration dictionary
    configuration_dict = configuration.dict()

    itemsizes = {}
    for key, dataset in configuration_dict["dataset_configurations"].items():
        itemsizes[key] = dataset["dtype"].itemsize

    serialized = json.loads(json.dumps(configuration_dict, default=custom_encoder))

    dataset_configurations = serialized["dataset_configurations"]  # Only provide dataset configurations

    for dataset in dataset_configurations.values():
        for key in PROPS_TO_REMOVE:
            del dataset[key]

    schema = list(configuration.schema()["$defs"].values())[0]
    for key in PROPS_TO_REMOVE:
        existed = schema["properties"].pop(key, None)  # Why is dtype not included but the rest are?
        if existed:
            schema["required"].remove(key)

    return dict(results=dataset_configurations, schema=schema, backend=backend, itemsizes=itemsizes)


def get_conversion_path_info(info: dict) -> dict:
    """Function used to resolve the path details for the conversion."""

    nwbfile_path = Path(info["nwbfile_path"])
    custom_output_directory = info.get("output_folder")
    project_name = info.get("project_name")
    run_stub_test = info.get("stub_test", False)
    default_output_base = STUB_SAVE_FOLDER_PATH if run_stub_test else CONVERSION_SAVE_FOLDER_PATH
    default_output_directory = default_output_base / project_name

    # add a subdirectory to a filepath if stub_test is true
    resolved_output_base = Path(custom_output_directory) if custom_output_directory else default_output_base
    resolved_output_directory = resolved_output_base / project_name
    resolved_output_path = resolved_output_directory / nwbfile_path

    return dict(file=resolved_output_path, directory=resolved_output_directory, default=default_output_directory)


def get_conversion_info(info: dict) -> dict:
    """Function used to organize the required information for conversion."""

    from neuroconv import NWBConverter

    path_info = get_conversion_path_info(info)
    resolved_output_path = path_info["file"]
    resolved_output_directory = path_info["directory"]
    default_output_directory = path_info["default"]

    # Remove symlink placed at the default_output_directory if this will hold real data
    if resolved_output_directory == default_output_directory and default_output_directory.is_symlink():
        default_output_directory.unlink()

    resolved_output_path.parent.mkdir(exist_ok=True, parents=True)  # Ensure all parent directories exist

    resolved_source_data = replace_none_with_nan(
        info["source_data"], resolve_references(get_custom_converter(info["interfaces"]).get_source_schema())
    )

    converter = instantiate_custom_converter(
        source_data=resolved_source_data,
        interface_class_dict=info["interfaces"],
        alignment_info=info.get("alignment", dict()),
    )

    # Ensure Ophys NaN values are resolved
    resolved_metadata = replace_none_with_nan(info["metadata"], resolve_references(converter.get_metadata_schema()))

    ecephys_metadata = resolved_metadata.get("Ecephys")

    if ecephys_metadata:

        # Quick fix to remove units
        has_units = "Units" in ecephys_metadata

        if has_units:

            ## NOTE: Currently do not allow editing units properties
            # shared_units_columns = ecephys_metadata["UnitColumns"]
            # for interface_name, interface_unit_results in ecephys_metadata["Units"].items():
            #     interface = converter.data_interface_objects[interface_name]

            #     update_sorting_properties_from_table_as_json(
            #         interface,
            #         unit_table_json=interface_unit_results,
            #         unit_column_info=shared_units_columns,
            #     )

            # ecephys_metadata["UnitProperties"] = [
            #     {"name": entry["name"], "description": entry["description"]} for entry in shared_units_columns
            # ]

            del ecephys_metadata["Units"]
            del ecephys_metadata["UnitColumns"]

        has_electrodes = "Electrodes" in ecephys_metadata
        if has_electrodes:

            shared_electrode_columns = ecephys_metadata["ElectrodeColumns"]

            for interface_name, interface_electrode_results in ecephys_metadata["Electrodes"].items():
                name_split = interface_name.split(" — ")

                if len(name_split) == 1:
                    sub_interface = name_split[0]
                elif len(name_split) == 2:
                    sub_interface, sub_sub_interface = name_split

                interface_or_subconverter = converter.data_interface_objects[sub_interface]

                if isinstance(interface_or_subconverter, NWBConverter):
                    subconverter = interface_or_subconverter

                    update_recording_properties_from_table_as_json(
                        recording_interface=subconverter.data_interface_objects[sub_sub_interface],
                        electrode_table_json=interface_electrode_results,
                        electrode_column_info=shared_electrode_columns,
                    )
                else:
                    interface = interface_or_subconverter

                    update_recording_properties_from_table_as_json(
                        recording_interface=interface,
                        electrode_table_json=interface_electrode_results,
                        electrode_column_info=shared_electrode_columns,
                    )

            ecephys_metadata["Electrodes"] = [
                {"name": entry["name"], "description": entry["description"]} for entry in shared_electrode_columns
            ]

            del ecephys_metadata["ElectrodeColumns"]

    # Correct timezone in metadata fields
    resolved_metadata["NWBFile"]["session_start_time"] = datetime.fromisoformat(
        resolved_metadata["NWBFile"]["session_start_time"]
    ).replace(tzinfo=zoneinfo.ZoneInfo(info["timezone"]))

    if "date_of_birth" in resolved_metadata["Subject"]:
        resolved_metadata["Subject"]["date_of_birth"] = datetime.fromisoformat(
            resolved_metadata["Subject"]["date_of_birth"]
        ).replace(tzinfo=zoneinfo.ZoneInfo(info["timezone"]))

    return (
        converter,
        resolved_metadata,
        path_info,
    )


def convert_to_nwb(
    info: dict,
    log_url: Optional[str] = None,
) -> str:
    """Function used to convert the source data to NWB format using the specified metadata."""

    path_info = get_conversion_path_info(info)
    output_path = path_info["file"]
    resolved_output_directory = path_info["directory"]
    default_output_directory = path_info["default"]

    create_file(info, log_url=log_url)

    # Create a symlink between the fake data and custom data
    if not resolved_output_directory == default_output_directory:
        if default_output_directory.exists():
            # If default default_output_directory is not a symlink, delete all contents and create a symlink there
            if not default_output_directory.is_symlink():
                rmtree(default_output_directory)

            # If the location is already a symlink, but points to a different output location
            # remove the existing symlink before creating a new one
            elif (
                default_output_directory.is_symlink()
                and default_output_directory.readlink() is not resolved_output_directory
            ):
                default_output_directory.unlink()

        # Create a pointer to the actual conversion outputs
        if not default_output_directory.exists():
            os.symlink(resolved_output_directory, default_output_directory)

    return dict(file=str(output_path))


def convert_all_to_nwb(
    url: str,
    files: List[dict],
    request_id: Optional[str],
    max_workers: int = 1,
    log_url: Optional[str] = None,
) -> List[str]:

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm_publisher import TQDMProgressSubscriber

    def on_progress_update(message):
        message["progress_bar_id"] = request_id  # Ensure request_id matches
        progress_handler.announce(
            dict(
                request_id=request_id,
                **message,
            )
        )

    futures = []
    file_paths = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for file_info in files:

            futures.append(
                executor.submit(
                    convert_to_nwb,
                    dict(
                        url=url,
                        request_id=request_id,
                        **file_info,
                    ),
                    log_url=log_url,
                )
            )

        inspection_iterable = TQDMProgressSubscriber(
            iterable=as_completed(futures),
            desc="Total files converted",
            total=len(futures),
            mininterval=0,
            on_progress_update=on_progress_update,
        )

        for future in inspection_iterable:
            output_filepath = future.result()
            file_paths.append(output_filepath)

        return file_paths


def upload_multiple_filesystem_objects_to_dandi(**kwargs) -> list[Path]:
    tmp_folder_path = _aggregate_symlinks_in_new_directory(kwargs["filesystem_paths"], "upload")
    innerKwargs = {**kwargs}
    del innerKwargs["filesystem_paths"]
    innerKwargs["nwb_folder_path"] = tmp_folder_path
    results = upload_folder_to_dandi(**innerKwargs)

    rmtree(tmp_folder_path, ignore_errors=True)

    return results


def upload_folder_to_dandi(
    dandiset_id: str,
    api_key: str,
    nwb_folder_path: Optional[str] = None,
    staging: Optional[bool] = None,  # Override default staging=True
    cleanup: Optional[bool] = None,
    number_of_jobs: Optional[int] = None,
    number_of_threads: Optional[int] = None,
    ignore_cache: bool = False,
) -> list[Path]:
    from neuroconv.tools.data_transfers import automatic_dandi_upload

    os.environ["DANDI_API_KEY"] = api_key  # Update API Key

    if ignore_cache:
        os.environ["DANDI_CACHE"] = "ignore"
    else:
        os.environ["DANDI_CACHE"] = ""

    return automatic_dandi_upload(
        dandiset_id=dandiset_id,
        nwb_folder_path=Path(nwb_folder_path),
        staging=staging,
        cleanup=cleanup,
        number_of_jobs=number_of_jobs or 1,
        number_of_threads=number_of_threads or 1,
    )


def upload_project_to_dandi(
    dandiset_id: str,
    api_key: str,
    project: Optional[str] = None,
    staging: Optional[bool] = None,  # Override default staging=True
    cleanup: Optional[bool] = None,
    number_of_jobs: Optional[int] = None,
    number_of_threads: Optional[int] = None,
    ignore_cache: bool = False,
) -> list[Path]:
    from neuroconv.tools.data_transfers import automatic_dandi_upload

    # CONVERSION_SAVE_FOLDER_PATH.mkdir(exist_ok=True, parents=True)  # Ensure base directory exists

    os.environ["DANDI_API_KEY"] = api_key  # Update API Key

    if ignore_cache:
        os.environ["DANDI_CACHE"] = "ignore"
    else:
        os.environ["DANDI_CACHE"] = ""

    return automatic_dandi_upload(
        dandiset_id=dandiset_id,
        nwb_folder_path=CONVERSION_SAVE_FOLDER_PATH / project,  # Scope valid DANDI upload paths to GUIDE projects
        staging=staging,
        cleanup=cleanup,
        number_of_jobs=number_of_jobs,
        number_of_threads=number_of_threads,
    )


# Create an events endpoint
def listen_to_neuroconv_progress_events():
    messages = progress_handler.listen()  # returns a queue.Queue
    while True:
        msg = messages.get()  # blocks until a new message arrives
        yield format_sse(msg)


def generate_dataset(input_path: str, output_path: str) -> dict:
    base_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists():
        rmtree(output_path)

    subjects = ["mouse1", "mouse2"]

    sessions = ["Session1", "Session2"]

    base_id = "Session1"

    for subject in subjects:
        for session in sessions:
            full_id = f"{subject}_{session}"
            session_output_directory = output_path / subject / full_id
            spikeglx_base_directory = base_path / "spikeglx" / f"{base_id}_g0"
            phy_base_directory = base_path / "phy"

            spikeglx_output_dir = session_output_directory / f"{full_id}_g0"
            phy_output_dir = session_output_directory / f"{full_id}_phy"

            copytree(spikeglx_base_directory, spikeglx_output_dir)

            # Rename directories
            for root, dirs, files in os.walk(spikeglx_output_dir):
                for dir in dirs:
                    if base_id in dir:
                        os.rename(os.path.join(root, dir), os.path.join(root, dir.replace(base_id, full_id)))

            # Rename files
            for root, dirs, files in os.walk(spikeglx_output_dir):
                for file in files:
                    if base_id in file:
                        os.rename(os.path.join(root, file), os.path.join(root, file.replace(base_id, full_id)))

            copytree(phy_base_directory, phy_output_dir)

    return {"output_path": str(output_path)}


def _inspect_file_per_job(
    nwbfile_path: str,
    url,
    ignore: Optional[List[str]] = None,
    request_id: Optional[str] = None,
) -> list:

    import nwbinspector
    import requests
    from pynwb import NWBHDF5IO
    from tqdm_publisher import TQDMProgressSubscriber

    checks = nwbinspector.configure_checks(
        checks=nwbinspector.available_checks,
        config=nwbinspector.load_config(filepath_or_keyword="dandi"),
        ignore=ignore,
    )

    progress_bar_options = dict(
        mininterval=0,
        on_progress_update=lambda message: requests.post(url=url, json=dict(request_id=request_id, **message)),
    )

    with NWBHDF5IO(path=nwbfile_path, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        messages = list(
            nwbinspector.run_checks(
                nwbfile=nwbfile,
                checks=checks,
                progress_bar_class=TQDMProgressSubscriber,
                progress_bar_options=progress_bar_options,
            )
        )
        for message in messages:
            if message.file_path is None:
                message.file_path = nwbfile_path  # Add file path to message if it is missing

        return messages


def _inspect_all(url, config):

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from nwbinspector.utils import calculate_number_of_cpu
    from tqdm_publisher import TQDMProgressSubscriber

    path = config.pop("path", None)

    paths = [path] if path else config.pop("paths", [])

    nwbfile_paths = []
    for path in paths:
        posix_path = Path(path)
        if posix_path.is_file():
            nwbfile_paths.append(posix_path)
        else:
            nwbfile_paths.extend(list(posix_path.rglob("*.nwb")))

    request_id = config.pop("request_id", None)

    n_jobs = config.get("n_jobs", -2)  # Default to all but one CPU
    n_jobs = calculate_number_of_cpu(requested_cpu=n_jobs)
    n_jobs = None if n_jobs == -1 else n_jobs

    futures = list()

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for nwbfile_path in nwbfile_paths:
            futures.append(
                executor.submit(
                    _inspect_file_per_job,
                    nwbfile_path=str(nwbfile_path),
                    ignore=config.get("ignore"),
                    url=url,
                    request_id=request_id,
                )
            )

        messages = list()

        # Announce directly
        def on_progress_update(message):
            message["progress_bar_id"] = request_id  # Ensure request_id matches
            progress_handler.announce(
                dict(
                    request_id=request_id,
                    **message,
                )
            )

        inspection_iterable = TQDMProgressSubscriber(
            iterable=as_completed(futures),
            desc="Total files inspected",
            total=len(futures),
            mininterval=0,
            on_progress_update=on_progress_update,
        )

        i = 0
        for future in inspection_iterable:
            i += 1
            for message in future.result():
                messages.append(message)

    return messages


def inspect_all(url, payload) -> dict:
    from pickle import PicklingError

    import nwbinspector

    try:
        messages = _inspect_all(url, payload)
    except PicklingError as exception:
        if "attribute lookup auto_parse_some_output on nwbinspector.register_checks failed" in str(exception):
            del payload["n_jobs"]
            messages = _inspect_all(url, payload)
        else:
            raise exception
    except Exception as exception:
        raise exception

    header = nwbinspector._formatting._get_report_header()
    header["NWBInspector_version"] = str(header["NWBInspector_version"])
    json_report = dict(
        header=header, messages=messages, text="\n".join(nwbinspector.format_messages(messages=messages))
    )

    return json.loads(json.dumps(obj=json_report, cls=nwbinspector.InspectorOutputJSONEncoder))


def _aggregate_symlinks_in_new_directory(paths, reason="", folder_path=None) -> Path:
    if folder_path is None:
        folder_path = GUIDE_ROOT_FOLDER / ".temp" / reason / f"temp_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    folder_path.mkdir(parents=True)

    for path in paths:
        path = Path(path)
        new_path = folder_path / path.name
        if path.is_dir():
            _aggregate_symlinks_in_new_directory(
                list(map(lambda name: os.path.join(path, name), os.listdir(path))), None, new_path
            )
        else:
            new_path.symlink_to(path, path.is_dir())

    return folder_path


def _format_spikeglx_meta_file(bin_file_path: str) -> str:
    bin_file_path = Path(bin_file_path)

    with open(file=bin_file_path, mode="rb") as io:
        file_sha1 = hashlib.sha1(string=io.read()).hexdigest().upper()
    file_size = bin_file_path.stat().st_size
    file_time_seconds = file_size / (385 * 2)  # 385 channels with int16 itemsize

    meta_structure = f"""acqApLfSy=384,384,1
appVersion=20190327
fileCreateTime={(datetime.now() - timedelta(hours=24)).isoformat(timespec='seconds')}
fileName={bin_file_path}
fileSHA1={file_sha1}
fileSizeBytes={file_size}
fileTimeSecs={file_time_seconds}
firstSample=0
gateMode=Immediate
imAiRangeMax=0.6
imAiRangeMin=-0.6
imCalibrated=true
imDatApi=1.15
imDatBs_fw=1.1.128
imDatBsc_fw=1.0.151
imDatBsc_hw=2.1
imDatBsc_pn=NP2_QBSC_00
imDatBsc_sn=434
imDatFx_hw=1.2
imDatFx_pn=NP2_FLEX_0
imDatHs_fw=5.1
imDatHs_pn=NP2_HS_30
imDatHs_sn=1116
imDatPrb_pn=PRB_1_4_0480_1
imDatPrb_port=2
imDatPrb_slot=2
imDatPrb_sn=18194809281
imDatPrb_type=0
imLEDEnable=false
imRoFile=C:/fake_windows_path/Gain_ap500_lfp125.imro
imSampRate=30000
imStdby=
imTrgRising=true
imTrgSource=0
nSavedChans=385
snsApLfSy=384,0,1
snsSaveChanSubset=0:383,768
syncImInputSlot=2
syncSourceIdx=0
syncSourcePeriod=1
trigMode=Immediate
typeImEnabled=1
typeNiEnabled=1
typeThis=imec
userNotes=
~imroTbl=(0,384)(0 0 0 500 125 1)(1 0 0 500 125 1)(2 0 0 500 125 1)(3 0 0 500 125 1)(4 0 0 500 125 1)(5 0 0 500 125 1)(6 0 0 500 125 1)(7 0 0 500 125 1)(8 0 0 500 125 1)(9 0 0 500 125 1)(10 0 0 500 125 1)(11 0 0 500 125 1)(12 0 0 500 125 1)(13 0 0 500 125 1)(14 0 0 500 125 1)(15 0 0 500 125 1)(16 0 0 500 125 1)(17 0 0 500 125 1)(18 0 0 500 125 1)(19 0 0 500 125 1)(20 0 0 500 125 1)(21 0 0 500 125 1)(22 0 0 500 125 1)(23 0 0 500 125 1)(24 0 0 500 125 1)(25 0 0 500 125 1)(26 0 0 500 125 1)(27 0 0 500 125 1)(28 0 0 500 125 1)(29 0 0 500 125 1)(30 0 0 500 125 1)(31 0 0 500 125 1)(32 0 0 500 125 1)(33 0 0 500 125 1)(34 0 0 500 125 1)(35 0 0 500 125 1)(36 0 0 500 125 1)(37 0 0 500 125 1)(38 0 0 500 125 1)(39 0 0 500 125 1)(40 0 0 500 125 1)(41 0 0 500 125 1)(42 0 0 500 125 1)(43 0 0 500 125 1)(44 0 0 500 125 1)(45 0 0 500 125 1)(46 0 0 500 125 1)(47 0 0 500 125 1)(48 0 0 500 125 1)(49 0 0 500 125 1)(50 0 0 500 125 1)(51 0 0 500 125 1)(52 0 0 500 125 1)(53 0 0 500 125 1)(54 0 0 500 125 1)(55 0 0 500 125 1)(56 0 0 500 125 1)(57 0 0 500 125 1)(58 0 0 500 125 1)(59 0 0 500 125 1)(60 0 0 500 125 1)(61 0 0 500 125 1)(62 0 0 500 125 1)(63 0 0 500 125 1)(64 0 0 500 125 1)(65 0 0 500 125 1)(66 0 0 500 125 1)(67 0 0 500 125 1)(68 0 0 500 125 1)(69 0 0 500 125 1)(70 0 0 500 125 1)(71 0 0 500 125 1)(72 0 0 500 125 1)(73 0 0 500 125 1)(74 0 0 500 125 1)(75 0 0 500 125 1)(76 0 0 500 125 1)(77 0 0 500 125 1)(78 0 0 500 125 1)(79 0 0 500 125 1)(80 0 0 500 125 1)(81 0 0 500 125 1)(82 0 0 500 125 1)(83 0 0 500 125 1)(84 0 0 500 125 1)(85 0 0 500 125 1)(86 0 0 500 125 1)(87 0 0 500 125 1)(88 0 0 500 125 1)(89 0 0 500 125 1)(90 0 0 500 125 1)(91 0 0 500 125 1)(92 0 0 500 125 1)(93 0 0 500 125 1)(94 0 0 500 125 1)(95 0 0 500 125 1)(96 0 0 500 125 1)(97 0 0 500 125 1)(98 0 0 500 125 1)(99 0 0 500 125 1)(100 0 0 500 125 1)(101 0 0 500 125 1)(102 0 0 500 125 1)(103 0 0 500 125 1)(104 0 0 500 125 1)(105 0 0 500 125 1)(106 0 0 500 125 1)(107 0 0 500 125 1)(108 0 0 500 125 1)(109 0 0 500 125 1)(110 0 0 500 125 1)(111 0 0 500 125 1)(112 0 0 500 125 1)(113 0 0 500 125 1)(114 0 0 500 125 1)(115 0 0 500 125 1)(116 0 0 500 125 1)(117 0 0 500 125 1)(118 0 0 500 125 1)(119 0 0 500 125 1)(120 0 0 500 125 1)(121 0 0 500 125 1)(122 0 0 500 125 1)(123 0 0 500 125 1)(124 0 0 500 125 1)(125 0 0 500 125 1)(126 0 0 500 125 1)(127 0 0 500 125 1)(128 0 0 500 125 1)(129 0 0 500 125 1)(130 0 0 500 125 1)(131 0 0 500 125 1)(132 0 0 500 125 1)(133 0 0 500 125 1)(134 0 0 500 125 1)(135 0 0 500 125 1)(136 0 0 500 125 1)(137 0 0 500 125 1)(138 0 0 500 125 1)(139 0 0 500 125 1)(140 0 0 500 125 1)(141 0 0 500 125 1)(142 0 0 500 125 1)(143 0 0 500 125 1)(144 0 0 500 125 1)(145 0 0 500 125 1)(146 0 0 500 125 1)(147 0 0 500 125 1)(148 0 0 500 125 1)(149 0 0 500 125 1)(150 0 0 500 125 1)(151 0 0 500 125 1)(152 0 0 500 125 1)(153 0 0 500 125 1)(154 0 0 500 125 1)(155 0 0 500 125 1)(156 0 0 500 125 1)(157 0 0 500 125 1)(158 0 0 500 125 1)(159 0 0 500 125 1)(160 0 0 500 125 1)(161 0 0 500 125 1)(162 0 0 500 125 1)(163 0 0 500 125 1)(164 0 0 500 125 1)(165 0 0 500 125 1)(166 0 0 500 125 1)(167 0 0 500 125 1)(168 0 0 500 125 1)(169 0 0 500 125 1)(170 0 0 500 125 1)(171 0 0 500 125 1)(172 0 0 500 125 1)(173 0 0 500 125 1)(174 0 0 500 125 1)(175 0 0 500 125 1)(176 0 0 500 125 1)(177 0 0 500 125 1)(178 0 0 500 125 1)(179 0 0 500 125 1)(180 0 0 500 125 1)(181 0 0 500 125 1)(182 0 0 500 125 1)(183 0 0 500 125 1)(184 0 0 500 125 1)(185 0 0 500 125 1)(186 0 0 500 125 1)(187 0 0 500 125 1)(188 0 0 500 125 1)(189 0 0 500 125 1)(190 0 0 500 125 1)(191 0 0 500 125 1)(192 0 0 500 125 1)(193 0 0 500 125 1)(194 0 0 500 125 1)(195 0 0 500 125 1)(196 0 0 500 125 1)(197 0 0 500 125 1)(198 0 0 500 125 1)(199 0 0 500 125 1)(200 0 0 500 125 1)(201 0 0 500 125 1)(202 0 0 500 125 1)(203 0 0 500 125 1)(204 0 0 500 125 1)(205 0 0 500 125 1)(206 0 0 500 125 1)(207 0 0 500 125 1)(208 0 0 500 125 1)(209 0 0 500 125 1)(210 0 0 500 125 1)(211 0 0 500 125 1)(212 0 0 500 125 1)(213 0 0 500 125 1)(214 0 0 500 125 1)(215 0 0 500 125 1)(216 0 0 500 125 1)(217 0 0 500 125 1)(218 0 0 500 125 1)(219 0 0 500 125 1)(220 0 0 500 125 1)(221 0 0 500 125 1)(222 0 0 500 125 1)(223 0 0 500 125 1)(224 0 0 500 125 1)(225 0 0 500 125 1)(226 0 0 500 125 1)(227 0 0 500 125 1)(228 0 0 500 125 1)(229 0 0 500 125 1)(230 0 0 500 125 1)(231 0 0 500 125 1)(232 0 0 500 125 1)(233 0 0 500 125 1)(234 0 0 500 125 1)(235 0 0 500 125 1)(236 0 0 500 125 1)(237 0 0 500 125 1)(238 0 0 500 125 1)(239 0 0 500 125 1)(240 0 0 500 125 1)(241 0 0 500 125 1)(242 0 0 500 125 1)(243 0 0 500 125 1)(244 0 0 500 125 1)(245 0 0 500 125 1)(246 0 0 500 125 1)(247 0 0 500 125 1)(248 0 0 500 125 1)(249 0 0 500 125 1)(250 0 0 500 125 1)(251 0 0 500 125 1)(252 0 0 500 125 1)(253 0 0 500 125 1)(254 0 0 500 125 1)(255 0 0 500 125 1)(256 0 0 500 125 1)(257 0 0 500 125 1)(258 0 0 500 125 1)(259 0 0 500 125 1)(260 0 0 500 125 1)(261 0 0 500 125 1)(262 0 0 500 125 1)(263 0 0 500 125 1)(264 0 0 500 125 1)(265 0 0 500 125 1)(266 0 0 500 125 1)(267 0 0 500 125 1)(268 0 0 500 125 1)(269 0 0 500 125 1)(270 0 0 500 125 1)(271 0 0 500 125 1)(272 0 0 500 125 1)(273 0 0 500 125 1)(274 0 0 500 125 1)(275 0 0 500 125 1)(276 0 0 500 125 1)(277 0 0 500 125 1)(278 0 0 500 125 1)(279 0 0 500 125 1)(280 0 0 500 125 1)(281 0 0 500 125 1)(282 0 0 500 125 1)(283 0 0 500 125 1)(284 0 0 500 125 1)(285 0 0 500 125 1)(286 0 0 500 125 1)(287 0 0 500 125 1)(288 0 0 500 125 1)(289 0 0 500 125 1)(290 0 0 500 125 1)(291 0 0 500 125 1)(292 0 0 500 125 1)(293 0 0 500 125 1)(294 0 0 500 125 1)(295 0 0 500 125 1)(296 0 0 500 125 1)(297 0 0 500 125 1)(298 0 0 500 125 1)(299 0 0 500 125 1)(300 0 0 500 125 1)(301 0 0 500 125 1)(302 0 0 500 125 1)(303 0 0 500 125 1)(304 0 0 500 125 1)(305 0 0 500 125 1)(306 0 0 500 125 1)(307 0 0 500 125 1)(308 0 0 500 125 1)(309 0 0 500 125 1)(310 0 0 500 125 1)(311 0 0 500 125 1)(312 0 0 500 125 1)(313 0 0 500 125 1)(314 0 0 500 125 1)(315 0 0 500 125 1)(316 0 0 500 125 1)(317 0 0 500 125 1)(318 0 0 500 125 1)(319 0 0 500 125 1)(320 0 0 500 125 1)(321 0 0 500 125 1)(322 0 0 500 125 1)(323 0 0 500 125 1)(324 0 0 500 125 1)(325 0 0 500 125 1)(326 0 0 500 125 1)(327 0 0 500 125 1)(328 0 0 500 125 1)(329 0 0 500 125 1)(330 0 0 500 125 1)(331 0 0 500 125 1)(332 0 0 500 125 1)(333 0 0 500 125 1)(334 0 0 500 125 1)(335 0 0 500 125 1)(336 0 0 500 125 1)(337 0 0 500 125 1)(338 0 0 500 125 1)(339 0 0 500 125 1)(340 0 0 500 125 1)(341 0 0 500 125 1)(342 0 0 500 125 1)(343 0 0 500 125 1)(344 0 0 500 125 1)(345 0 0 500 125 1)(346 0 0 500 125 1)(347 0 0 500 125 1)(348 0 0 500 125 1)(349 0 0 500 125 1)(350 0 0 500 125 1)(351 0 0 500 125 1)(352 0 0 500 125 1)(353 0 0 500 125 1)(354 0 0 500 125 1)(355 0 0 500 125 1)(356 0 0 500 125 1)(357 0 0 500 125 1)(358 0 0 500 125 1)(359 0 0 500 125 1)(360 0 0 500 125 1)(361 0 0 500 125 1)(362 0 0 500 125 1)(363 0 0 500 125 1)(364 0 0 500 125 1)(365 0 0 500 125 1)(366 0 0 500 125 1)(367 0 0 500 125 1)(368 0 0 500 125 1)(369 0 0 500 125 1)(370 0 0 500 125 1)(371 0 0 500 125 1)(372 0 0 500 125 1)(373 0 0 500 125 1)(374 0 0 500 125 1)(375 0 0 500 125 1)(376 0 0 500 125 1)(377 0 0 500 125 1)(378 0 0 500 125 1)(379 0 0 500 125 1)(380 0 0 500 125 1)(381 0 0 500 125 1)(382 0 0 500 125 1)(383 0 0 500 125 1)
~snsChanMap=(384,384,1)(AP0;0:0)(AP1;1:1)(AP2;2:2)(AP3;3:3)(AP4;4:4)(AP5;5:5)(AP6;6:6)(AP7;7:7)(AP8;8:8)(AP9;9:9)(AP10;10:10)(AP11;11:11)(AP12;12:12)(AP13;13:13)(AP14;14:14)(AP15;15:15)(AP16;16:16)(AP17;17:17)(AP18;18:18)(AP19;19:19)(AP20;20:20)(AP21;21:21)(AP22;22:22)(AP23;23:23)(AP24;24:24)(AP25;25:25)(AP26;26:26)(AP27;27:27)(AP28;28:28)(AP29;29:29)(AP30;30:30)(AP31;31:31)(AP32;32:32)(AP33;33:33)(AP34;34:34)(AP35;35:35)(AP36;36:36)(AP37;37:37)(AP38;38:38)(AP39;39:39)(AP40;40:40)(AP41;41:41)(AP42;42:42)(AP43;43:43)(AP44;44:44)(AP45;45:45)(AP46;46:46)(AP47;47:47)(AP48;48:48)(AP49;49:49)(AP50;50:50)(AP51;51:51)(AP52;52:52)(AP53;53:53)(AP54;54:54)(AP55;55:55)(AP56;56:56)(AP57;57:57)(AP58;58:58)(AP59;59:59)(AP60;60:60)(AP61;61:61)(AP62;62:62)(AP63;63:63)(AP64;64:64)(AP65;65:65)(AP66;66:66)(AP67;67:67)(AP68;68:68)(AP69;69:69)(AP70;70:70)(AP71;71:71)(AP72;72:72)(AP73;73:73)(AP74;74:74)(AP75;75:75)(AP76;76:76)(AP77;77:77)(AP78;78:78)(AP79;79:79)(AP80;80:80)(AP81;81:81)(AP82;82:82)(AP83;83:83)(AP84;84:84)(AP85;85:85)(AP86;86:86)(AP87;87:87)(AP88;88:88)(AP89;89:89)(AP90;90:90)(AP91;91:91)(AP92;92:92)(AP93;93:93)(AP94;94:94)(AP95;95:95)(AP96;96:96)(AP97;97:97)(AP98;98:98)(AP99;99:99)(AP100;100:100)(AP101;101:101)(AP102;102:102)(AP103;103:103)(AP104;104:104)(AP105;105:105)(AP106;106:106)(AP107;107:107)(AP108;108:108)(AP109;109:109)(AP110;110:110)(AP111;111:111)(AP112;112:112)(AP113;113:113)(AP114;114:114)(AP115;115:115)(AP116;116:116)(AP117;117:117)(AP118;118:118)(AP119;119:119)(AP120;120:120)(AP121;121:121)(AP122;122:122)(AP123;123:123)(AP124;124:124)(AP125;125:125)(AP126;126:126)(AP127;127:127)(AP128;128:128)(AP129;129:129)(AP130;130:130)(AP131;131:131)(AP132;132:132)(AP133;133:133)(AP134;134:134)(AP135;135:135)(AP136;136:136)(AP137;137:137)(AP138;138:138)(AP139;139:139)(AP140;140:140)(AP141;141:141)(AP142;142:142)(AP143;143:143)(AP144;144:144)(AP145;145:145)(AP146;146:146)(AP147;147:147)(AP148;148:148)(AP149;149:149)(AP150;150:150)(AP151;151:151)(AP152;152:152)(AP153;153:153)(AP154;154:154)(AP155;155:155)(AP156;156:156)(AP157;157:157)(AP158;158:158)(AP159;159:159)(AP160;160:160)(AP161;161:161)(AP162;162:162)(AP163;163:163)(AP164;164:164)(AP165;165:165)(AP166;166:166)(AP167;167:167)(AP168;168:168)(AP169;169:169)(AP170;170:170)(AP171;171:171)(AP172;172:172)(AP173;173:173)(AP174;174:174)(AP175;175:175)(AP176;176:176)(AP177;177:177)(AP178;178:178)(AP179;179:179)(AP180;180:180)(AP181;181:181)(AP182;182:182)(AP183;183:183)(AP184;184:184)(AP185;185:185)(AP186;186:186)(AP187;187:187)(AP188;188:188)(AP189;189:189)(AP190;190:190)(AP191;191:191)(AP192;192:192)(AP193;193:193)(AP194;194:194)(AP195;195:195)(AP196;196:196)(AP197;197:197)(AP198;198:198)(AP199;199:199)(AP200;200:200)(AP201;201:201)(AP202;202:202)(AP203;203:203)(AP204;204:204)(AP205;205:205)(AP206;206:206)(AP207;207:207)(AP208;208:208)(AP209;209:209)(AP210;210:210)(AP211;211:211)(AP212;212:212)(AP213;213:213)(AP214;214:214)(AP215;215:215)(AP216;216:216)(AP217;217:217)(AP218;218:218)(AP219;219:219)(AP220;220:220)(AP221;221:221)(AP222;222:222)(AP223;223:223)(AP224;224:224)(AP225;225:225)(AP226;226:226)(AP227;227:227)(AP228;228:228)(AP229;229:229)(AP230;230:230)(AP231;231:231)(AP232;232:232)(AP233;233:233)(AP234;234:234)(AP235;235:235)(AP236;236:236)(AP237;237:237)(AP238;238:238)(AP239;239:239)(AP240;240:240)(AP241;241:241)(AP242;242:242)(AP243;243:243)(AP244;244:244)(AP245;245:245)(AP246;246:246)(AP247;247:247)(AP248;248:248)(AP249;249:249)(AP250;250:250)(AP251;251:251)(AP252;252:252)(AP253;253:253)(AP254;254:254)(AP255;255:255)(AP256;256:256)(AP257;257:257)(AP258;258:258)(AP259;259:259)(AP260;260:260)(AP261;261:261)(AP262;262:262)(AP263;263:263)(AP264;264:264)(AP265;265:265)(AP266;266:266)(AP267;267:267)(AP268;268:268)(AP269;269:269)(AP270;270:270)(AP271;271:271)(AP272;272:272)(AP273;273:273)(AP274;274:274)(AP275;275:275)(AP276;276:276)(AP277;277:277)(AP278;278:278)(AP279;279:279)(AP280;280:280)(AP281;281:281)(AP282;282:282)(AP283;283:283)(AP284;284:284)(AP285;285:285)(AP286;286:286)(AP287;287:287)(AP288;288:288)(AP289;289:289)(AP290;290:290)(AP291;291:291)(AP292;292:292)(AP293;293:293)(AP294;294:294)(AP295;295:295)(AP296;296:296)(AP297;297:297)(AP298;298:298)(AP299;299:299)(AP300;300:300)(AP301;301:301)(AP302;302:302)(AP303;303:303)(AP304;304:304)(AP305;305:305)(AP306;306:306)(AP307;307:307)(AP308;308:308)(AP309;309:309)(AP310;310:310)(AP311;311:311)(AP312;312:312)(AP313;313:313)(AP314;314:314)(AP315;315:315)(AP316;316:316)(AP317;317:317)(AP318;318:318)(AP319;319:319)(AP320;320:320)(AP321;321:321)(AP322;322:322)(AP323;323:323)(AP324;324:324)(AP325;325:325)(AP326;326:326)(AP327;327:327)(AP328;328:328)(AP329;329:329)(AP330;330:330)(AP331;331:331)(AP332;332:332)(AP333;333:333)(AP334;334:334)(AP335;335:335)(AP336;336:336)(AP337;337:337)(AP338;338:338)(AP339;339:339)(AP340;340:340)(AP341;341:341)(AP342;342:342)(AP343;343:343)(AP344;344:344)(AP345;345:345)(AP346;346:346)(AP347;347:347)(AP348;348:348)(AP349;349:349)(AP350;350:350)(AP351;351:351)(AP352;352:352)(AP353;353:353)(AP354;354:354)(AP355;355:355)(AP356;356:356)(AP357;357:357)(AP358;358:358)(AP359;359:359)(AP360;360:360)(AP361;361:361)(AP362;362:362)(AP363;363:363)(AP364;364:364)(AP365;365:365)(AP366;366:366)(AP367;367:367)(AP368;368:368)(AP369;369:369)(AP370;370:370)(AP371;371:371)(AP372;372:372)(AP373;373:373)(AP374;374:374)(AP375;375:375)(AP376;376:376)(AP377;377:377)(AP378;378:378)(AP379;379:379)(AP380;380:380)(AP381;381:381)(AP382;382:382)(AP383;383:383)(SY0;768:768)
~snsShankMap=(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:1)(0:0:2:1)(0:1:2:1)(0:0:3:1)(0:1:3:1)(0:0:4:1)(0:1:4:1)(0:0:5:1)(0:1:5:1)(0:0:6:1)(0:1:6:1)(0:0:7:1)(0:1:7:1)(0:0:8:1)(0:1:8:1)(0:0:9:1)(0:1:9:1)(0:0:10:1)(0:1:10:1)(0:0:11:1)(0:1:11:1)(0:0:12:1)(0:1:12:1)(0:0:13:1)(0:1:13:1)(0:0:14:1)(0:1:14:1)(0:0:15:1)(0:1:15:1)(0:0:16:1)(0:1:16:1)(0:0:17:1)(0:1:17:1)(0:0:18:1)(0:1:18:1)(0:0:19:1)(0:1:19:1)(0:0:20:1)(0:1:20:1)(0:0:21:1)(0:1:21:1)(0:0:22:1)(0:1:22:1)(0:0:23:1)(0:1:23:1)(0:0:24:1)(0:1:24:1)(0:0:25:1)(0:1:25:1)(0:0:26:1)(0:1:26:1)(0:0:27:1)(0:1:27:1)(0:0:28:1)(0:1:28:1)(0:0:29:1)(0:1:29:1)(0:0:30:1)(0:1:30:1)(0:0:31:1)(0:1:31:1)(0:0:32:1)(0:1:32:1)(0:0:33:1)(0:1:33:1)(0:0:34:1)(0:1:34:1)(0:0:35:1)(0:1:35:1)(0:0:36:1)(0:1:36:1)(0:0:37:1)(0:1:37:1)(0:0:38:1)(0:1:38:1)(0:0:39:1)(0:1:39:1)(0:0:40:1)(0:1:40:1)(0:0:41:1)(0:1:41:1)(0:0:42:1)(0:1:42:1)(0:0:43:1)(0:1:43:1)(0:0:44:1)(0:1:44:1)(0:0:45:1)(0:1:45:1)(0:0:46:1)(0:1:46:1)(0:0:47:1)(0:1:47:1)(0:0:48:1)(0:1:48:1)(0:0:49:1)(0:1:49:1)(0:0:50:1)(0:1:50:1)(0:0:51:1)(0:1:51:1)(0:0:52:1)(0:1:52:1)(0:0:53:1)(0:1:53:1)(0:0:54:1)(0:1:54:1)(0:0:55:1)(0:1:55:1)(0:0:56:1)(0:1:56:1)(0:0:57:1)(0:1:57:1)(0:0:58:1)(0:1:58:1)(0:0:59:1)(0:1:59:1)(0:0:60:1)(0:1:60:1)(0:0:61:1)(0:1:61:1)(0:0:62:1)(0:1:62:1)(0:0:63:1)(0:1:63:1)(0:0:64:1)(0:1:64:1)(0:0:65:1)(0:1:65:1)(0:0:66:1)(0:1:66:1)(0:0:67:1)(0:1:67:1)(0:0:68:1)(0:1:68:1)(0:0:69:1)(0:1:69:1)(0:0:70:1)(0:1:70:1)(0:0:71:1)(0:1:71:1)(0:0:72:1)(0:1:72:1)(0:0:73:1)(0:1:73:1)(0:0:74:1)(0:1:74:1)(0:0:75:1)(0:1:75:1)(0:0:76:1)(0:1:76:1)(0:0:77:1)(0:1:77:1)(0:0:78:1)(0:1:78:1)(0:0:79:1)(0:1:79:1)(0:0:80:1)(0:1:80:1)(0:0:81:1)(0:1:81:1)(0:0:82:1)(0:1:82:1)(0:0:83:1)(0:1:83:1)(0:0:84:1)(0:1:84:1)(0:0:85:1)(0:1:85:1)(0:0:86:1)(0:1:86:1)(0:0:87:1)(0:1:87:1)(0:0:88:1)(0:1:88:1)(0:0:89:1)(0:1:89:1)(0:0:90:1)(0:1:90:1)(0:0:91:1)(0:1:91:1)(0:0:92:1)(0:1:92:1)(0:0:93:1)(0:1:93:1)(0:0:94:1)(0:1:94:1)(0:0:95:1)(0:1:95:0)(0:0:96:1)(0:1:96:1)(0:0:97:1)(0:1:97:1)(0:0:98:1)(0:1:98:1)(0:0:99:1)(0:1:99:1)(0:0:100:1)(0:1:100:1)(0:0:101:1)(0:1:101:1)(0:0:102:1)(0:1:102:1)(0:0:103:1)(0:1:103:1)(0:0:104:1)(0:1:104:1)(0:0:105:1)(0:1:105:1)(0:0:106:1)(0:1:106:1)(0:0:107:1)(0:1:107:1)(0:0:108:1)(0:1:108:1)(0:0:109:1)(0:1:109:1)(0:0:110:1)(0:1:110:1)(0:0:111:1)(0:1:111:1)(0:0:112:1)(0:1:112:1)(0:0:113:1)(0:1:113:1)(0:0:114:1)(0:1:114:1)(0:0:115:1)(0:1:115:1)(0:0:116:1)(0:1:116:1)(0:0:117:1)(0:1:117:1)(0:0:118:1)(0:1:118:1)(0:0:119:1)(0:1:119:1)(0:0:120:1)(0:1:120:1)(0:0:121:1)(0:1:121:1)(0:0:122:1)(0:1:122:1)(0:0:123:1)(0:1:123:1)(0:0:124:1)(0:1:124:1)(0:0:125:1)(0:1:125:1)(0:0:126:1)(0:1:126:1)(0:0:127:1)(0:1:127:1)(0:0:128:1)(0:1:128:1)(0:0:129:1)(0:1:129:1)(0:0:130:1)(0:1:130:1)(0:0:131:1)(0:1:131:1)(0:0:132:1)(0:1:132:1)(0:0:133:1)(0:1:133:1)(0:0:134:1)(0:1:134:1)(0:0:135:1)(0:1:135:1)(0:0:136:1)(0:1:136:1)(0:0:137:1)(0:1:137:1)(0:0:138:1)(0:1:138:1)(0:0:139:1)(0:1:139:1)(0:0:140:1)(0:1:140:1)(0:0:141:1)(0:1:141:1)(0:0:142:1)(0:1:142:1)(0:0:143:1)(0:1:143:1)(0:0:144:1)(0:1:144:1)(0:0:145:1)(0:1:145:1)(0:0:146:1)(0:1:146:1)(0:0:147:1)(0:1:147:1)(0:0:148:1)(0:1:148:1)(0:0:149:1)(0:1:149:1)(0:0:150:1)(0:1:150:1)(0:0:151:1)(0:1:151:1)(0:0:152:1)(0:1:152:1)(0:0:153:1)(0:1:153:1)(0:0:154:1)(0:1:154:1)(0:0:155:1)(0:1:155:1)(0:0:156:1)(0:1:156:1)(0:0:157:1)(0:1:157:1)(0:0:158:1)(0:1:158:1)(0:0:159:1)(0:1:159:1)(0:0:160:1)(0:1:160:1)(0:0:161:1)(0:1:161:1)(0:0:162:1)(0:1:162:1)(0:0:163:1)(0:1:163:1)(0:0:164:1)(0:1:164:1)(0:0:165:1)(0:1:165:1)(0:0:166:1)(0:1:166:1)(0:0:167:1)(0:1:167:1)(0:0:168:1)(0:1:168:1)(0:0:169:1)(0:1:169:1)(0:0:170:1)(0:1:170:1)(0:0:171:1)(0:1:171:1)(0:0:172:1)(0:1:172:1)(0:0:173:1)(0:1:173:1)(0:0:174:1)(0:1:174:1)(0:0:175:1)(0:1:175:1)(0:0:176:1)(0:1:176:1)(0:0:177:1)(0:1:177:1)(0:0:178:1)(0:1:178:1)(0:0:179:1)(0:1:179:1)(0:0:180:1)(0:1:180:1)(0:0:181:1)(0:1:181:1)(0:0:182:1)(0:1:182:1)(0:0:183:1)(0:1:183:1)(0:0:184:1)(0:1:184:1)(0:0:185:1)(0:1:185:1)(0:0:186:1)(0:1:186:1)(0:0:187:1)(0:1:187:1)(0:0:188:1)(0:1:188:1)(0:0:189:1)(0:1:189:1)(0:0:190:1)(0:1:190:1)(0:0:191:1)(0:1:191:1)
    """
    return meta_structure


def generate_test_data(output_path: str):
    """
    Autogenerate the data formats needed for the tutorial pipeline.

    Consists of a single-probe single-segment SpikeGLX recording (both AP and LF bands) as well as Phy sorting data.
    """
    import spikeinterface
    import spikeinterface.exporters
    import spikeinterface.preprocessing

    spikeinterface.set_global_job_kwargs(n_jobs=-1)

    base_path = Path(output_path)
    spikeglx_output_folder = base_path / "spikeglx"
    phy_output_folder = base_path / "phy"

    # Define Neuropixels-like values for sampling rates and conversion factors
    duration_in_s = 3.0
    number_of_units = 50
    number_of_channels = 385  # Have to include 'sync' channel to be proper SpikeGLX. TODO: artificiate sync pulses
    conversion_factor_to_uV = 2.34375
    ap_sampling_frequency = 30_000.0
    lf_sampling_frequency = 2_500.0
    downsample_factor = int(ap_sampling_frequency / lf_sampling_frequency)

    # Generate synthetic sorting and voltage traces with waveforms around them
    artificial_ap_band_in_uV, sorting = spikeinterface.generate_ground_truth_recording(
        durations=[duration_in_s],
        sampling_frequency=ap_sampling_frequency,
        num_channels=number_of_channels,
        dtype="float32",
        num_units=number_of_units,
        seed=0,  # Fixed seed for reproducibility
    )

    unscaled_artificial_ap_band = spikeinterface.preprocessing.scale(
        recording=artificial_ap_band_in_uV, gain=1 / conversion_factor_to_uV
    )
    int16_artificial_ap_band = unscaled_artificial_ap_band.astype(dtype="int16")
    int16_artificial_ap_band.set_channel_gains(conversion_factor_to_uV)

    unscaled_artificial_lf_filter = spikeinterface.preprocessing.bandpass_filter(
        recording=unscaled_artificial_ap_band, freq_min=0.5, freq_max=1_000
    )
    unscaled_artificial_lf_band = spikeinterface.preprocessing.decimate(
        recording=unscaled_artificial_lf_filter, decimation_factor=downsample_factor
    )
    int16_artificial_lf_band = unscaled_artificial_lf_band.astype(dtype="int16")
    int16_artificial_lf_band.set_channel_gains(conversion_factor_to_uV)

    ap_file_path = spikeglx_output_folder / "Session1_g0" / "Session1_g0_imec0" / "Session1_g0_t0.imec0.ap.bin"
    ap_meta_file_path = spikeglx_output_folder / "Session1_g0" / "Session1_g0_imec0" / "Session1_g0_t0.imec0.ap.meta"
    lf_file_path = spikeglx_output_folder / "Session1_g0" / "Session1_g0_imec0" / "Session1_g0_t0.imec0.lf.bin"
    lf_meta_file_path = spikeglx_output_folder / "Session1_g0" / "Session1_g0_imec0" / "Session1_g0_t0.imec0.lf.meta"

    # Make .bin files
    ap_file_path.parent.mkdir(parents=True, exist_ok=True)
    spikeinterface.write_binary_recording(recording=int16_artificial_ap_band, file_paths=[ap_file_path])
    spikeinterface.write_binary_recording(recording=int16_artificial_lf_band, file_paths=[lf_file_path])

    # Make .meta files
    ap_meta_content = _format_spikeglx_meta_file(bin_file_path=ap_file_path)
    with open(file=ap_meta_file_path, mode="w") as io:
        io.write(ap_meta_content)

    lf_meta_content = _format_spikeglx_meta_file(bin_file_path=lf_file_path)
    with open(file=lf_meta_file_path, mode="w") as io:
        io.write(lf_meta_content)

    # Make Phy folder - see https://spikeinterface.readthedocs.io/en/latest/modules/exporters.html
    sorting_analyzer = spikeinterface.create_sorting_analyzer(
        sorting=sorting, recording=artificial_ap_band_in_uV, mode="memory", sparse=False
    )
    sorting_analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])
    sorting_analyzer.compute("spike_amplitudes")
    sorting_analyzer.compute("principal_components", n_components=5, mode="by_channel_local")

    spikeinterface.exporters.export_to_phy(
        sorting_analyzer=sorting_analyzer, output_folder=phy_output_folder, remove_if_exists=True, copy_binary=False
    )


def map_dtype(dtype: str) -> str:
    if "<U" in dtype:
        return "str"
    else:
        return dtype


def get_property_dtype(extractor, property_name: str, ids: list, extra_props: dict) -> str:
    if property_name in extra_props:
        dtype = extra_props[property_name]["data_type"]
    else:
        dtype = str(extractor.get_property(key=property_name, ids=ids).dtype)

    # return type(recording.get_property(key=property_name)[0]).__name__.replace("_", "")
    # return dtype
    return map_dtype(dtype)


# Ecephys Helper Functions
def get_recording_interface_properties(recording_interface) -> Dict[str, Any]:
    """A convenience function for uniformly excluding certain properties of the provided recording extractor."""
    property_names = list(recording_interface.recording_extractor.get_property_keys())

    properties = {
        property_name: recording_interface.recording_extractor.get_property(key=property_name)
        for property_name in property_names
        if property_name not in EXCLUDED_RECORDING_INTERFACE_PROPERTIES
    }

    for property_name in EXTRA_RECORDING_INTERFACE_PROPERTIES:
        if property_name not in properties:
            properties[property_name] = {}

    return properties


def get_sorting_interface_properties(sorting_interface) -> Dict[str, Any]:
    """A convenience function for uniformly excluding certain properties of the provided sorting extractor."""
    property_names = list(sorting_interface.sorting_extractor.get_property_keys())

    properties = {
        property_name: sorting_interface.sorting_extractor.get_property(key=property_name)
        for property_name in property_names
        if property_name not in EXCLUDED_SORTING_INTERFACE_PROPERTIES
    }

    for property_name in EXTRA_SORTING_INTERFACE_PROPERTIES:
        if property_name not in properties:
            properties[property_name] = {}

    return properties


def get_unit_columns_json(interface) -> List[Dict[str, Any]]:
    """A convenience function for collecting and organizing the properties of the underlying sorting extractor."""
    properties = get_sorting_interface_properties(interface)

    property_descriptions = dict(clu_id="The cluster ID for the unit", group_id="The group ID for the unit")
    property_data_types = dict()

    for property_name, property_info in SORTING_INTERFACE_PROPERTY_OVERRIDES.items():
        description = property_info.get("description", None)
        data_type = property_info.get("data_type", None)
        if description:
            property_descriptions[property_name] = description
        if data_type:
            property_data_types[property_name] = data_type

    sorting_extractor = interface.sorting_extractor
    unit_ids = sorting_extractor.get_unit_ids()

    unit_columns = [
        dict(
            name=property_name,
            description=property_descriptions.get(property_name, "No description."),
            data_type=property_data_types.get(
                property_name,
                get_property_dtype(
                    extractor=sorting_extractor,
                    property_name=property_name,
                    ids=[unit_ids[0]],
                    extra_props=SORTING_INTERFACE_PROPERTY_OVERRIDES,
                ),
            ),
        )
        for property_name in properties.keys()
    ]

    return json.loads(json.dumps(obj=unit_columns))


def get_unit_table_json(interface) -> List[Dict[str, Any]]:
    """
    A convenience function for collecting and organizing the property values of the underlying sorting extractor.
    """

    from neuroconv.utils import NWBMetaDataEncoder

    sorting = interface.sorting_extractor

    properties = get_sorting_interface_properties(interface)

    unit_ids = sorting.get_unit_ids()

    table = list()
    for unit_id in unit_ids:

        unit_column = dict()

        for property_name in properties:

            if property_name == "unit_id":
                sorting_property_value = str(unit_id)  # Insert unit_id to view

            # elif property_name == "unit_name":
            #     sorting_property_value = str(unit_id) # By default, unit_name is unit_id (str)

            elif property_name in SORTING_INTERFACE_PROPERTY_OVERRIDES:
                try:
                    sorting_property_value = SORTING_INTERFACE_PROPERTY_OVERRIDES[property_name][
                        "default"
                    ]  # Get default value
                except:
                    sorting_property_value = sorting.get_property(key=property_name, ids=[unit_id])[0]
            else:
                sorting_property_value = sorting.get_property(key=property_name, ids=[unit_id])[
                    0  # First axis is always units in SI
                ]  # Since only fetching one unit at a time, use trivial zero-index
            unit_column.update({property_name: sorting_property_value})
        table.append(unit_column)
    table_as_json = json.loads(json.dumps(table, cls=NWBMetaDataEncoder))

    return table_as_json


def get_electrode_columns_json(interface) -> List[Dict[str, Any]]:
    """A convenience function for collecting and organizing the properties of the underlying recording extractor."""
    properties = get_recording_interface_properties(interface)

    # Hardcuded for SpikeGLX (NOTE: Update for more interfaces)
    property_descriptions = dict(
        channel_name="The name of this channel.",
        group_name="The name of the ElectrodeGroup this channel's electrode is a part of.",
        shank_electrode_number="0-based index of the electrode on the shank.",
        contact_shapes="The shape of the electrode.",
        inter_sample_shift="Time-delay of each channel sampling in proportion to the per-frame sampling period.",
        gain_to_uV="The scaling factor from the data type to microVolts, applied before the offset.",
        offset_to_uV="The offset from the data type to microVolts, applied after the gain.",
    )

    for property_name, property_info in RECORDING_INTERFACE_PROPERTY_OVERRIDES.items():
        description = property_info.get("description", None)
        if description:
            property_descriptions[property_name] = description

    # default_column_metadata =  interface.get_metadata()["Ecephys"]["ElectrodeColumns"]["properties"] # NOTE: This doesn't exist...
    # property_descriptions = {column_name: column_fields["description"] for column_name, column_fields in default_column_metadata}

    recording_extractor = interface.recording_extractor
    channel_ids = recording_extractor.get_channel_ids()

    electrode_columns = [
        dict(
            name=property_name,
            description=property_descriptions.get(property_name, "No description."),
            data_type=get_property_dtype(
                extractor=recording_extractor,
                property_name=property_name,
                ids=[channel_ids[0]],
                extra_props=RECORDING_INTERFACE_PROPERTY_OVERRIDES,
            ),
        )
        for property_name in properties.keys()
    ]

    # TODO: uncomment when neuroconv supports contact vectors (probe interface)
    # contact_vector = properties.pop("contact_vector", None)
    # if contact_vector is None:
    #     return json.loads(json.dumps(obj=electrode_columns))
    # # Unpack contact vector
    # for property_name in contact_vector.dtype.names:
    #     electrode_columns.append(
    #         dict(
    #             name=property_name,
    #             description=property_descriptions.get(property_name, ""),
    #             data_type=str(contact_vector.dtype.fields[property_name][0]),
    #         )
    #     )

    return json.loads(json.dumps(obj=electrode_columns))


def get_electrode_table_json(interface) -> List[Dict[str, Any]]:
    """
    A convenience function for collecting and organizing the property values of the underlying recording extractor.
    """

    from neuroconv.utils import NWBMetaDataEncoder

    recording = interface.recording_extractor

    properties = get_recording_interface_properties(interface)

    electrode_ids = recording.get_channel_ids()

    table = list()
    for electrode_id in electrode_ids:
        electrode_column = dict()
        for property_name in properties:
            if property_name in RECORDING_INTERFACE_PROPERTY_OVERRIDES:
                try:
                    recording_property_value = RECORDING_INTERFACE_PROPERTY_OVERRIDES[property_name][
                        "default"
                    ]  # Get default value
                except:
                    recording_property_value = recording.get_property(key=property_name, ids=[electrode_id])[0]
            else:
                recording_property_value = recording.get_property(key=property_name, ids=[electrode_id])[
                    0  # First axis is always electodes in SI
                ]  # Since only fetching one electrode at a time, use trivial zero-index
            electrode_column.update({property_name: recording_property_value})
        table.append(electrode_column)
    table_as_json = json.loads(json.dumps(table, cls=NWBMetaDataEncoder))

    return table_as_json


def update_recording_properties_from_table_as_json(
    recording_interface, electrode_column_info: dict, electrode_table_json: List[Dict[str, Any]]
):
    import numpy as np

    # # Extract contact vector properties
    properties = get_recording_interface_properties(recording_interface)

    # TODO: uncomment and adapt when neuroconv supports contact vectors (probe interface)
    # contact_vector = properties.pop("contact_vector", None)
    # contact_vector_dtypes = {}
    # if contact_vector is not None:
    #     # Remove names from contact vector from the electrode_column_info and add to reconstructed_contact_vector_info
    #     contact_vector_dtypes = contact_vector.dtype
    #     # contact_vector_dtypes = { property_name: next((item for item in electrode_column_info if item['name'] == property_name), None)["data_type"] for property_name in contact_vector.dtype.names}
    #     # Remove contact vector properties from electrode_column_info
    #     for property_name in contact_vector.dtype.names:
    #         found = next((item for item in electrode_column_info if item["name"] == property_name), None)
    #         if found:
    #             electrode_column_info.remove(found)

    # Organize dtypes
    electrode_column_data_types = {column["name"]: column["data_type"] for column in electrode_column_info}
    # electrode_column_data_types["contact_vector"] = contact_vector_dtypes  # Provide contact vector information

    recording_extractor = recording_interface.recording_extractor
    channel_ids = recording_extractor.get_channel_ids()

    # TODO: uncomment when neuroconv supports contact vectors (probe interface)
    # property_names = recording_extractor.get_property_keys()
    # if "contact_vector" in property_names:
    #     modified_contact_vector = np.array(recording_extractor.get_property(key="contact_vector"))  # copy
    #     contact_vector_property_names = list(modified_contact_vector.dtype.names)

    for entry_index, entry in enumerate(electrode_table_json):
        electrode_properties = dict(entry)  # copy
        # channel_name = electrode_properties.pop("channel_name", None)
        for property_name, property_value in electrode_properties.items():
            if property_name not in electrode_column_data_types:  # Skip data with missing column information
                continue
            # TODO: uncomment when neuroconv supports contact vectors (probe interface)
            # elif property_name in contact_vector_property_names:
            #     property_index = contact_vector_property_names.index(property_name)
            #     modified_contact_vector[entry_index][property_index] = property_value
            else:
                recording_extractor.set_property(
                    key=property_name,
                    values=np.array([property_value], dtype=electrode_column_data_types[property_name]),
                    ids=[channel_ids[entry_index]],  # Assume rows match indices of channel list
                )

    # TODO: uncomment when neuroconv supports contact vectors (probe interface)
    # if "contact_vector" in property_names:
    #     recording_extractor.set_property(key="contact_vector", values=modified_contact_vector)


def update_sorting_properties_from_table_as_json(
    sorting_interface, unit_column_info: dict, unit_table_json: List[Dict[str, Any]]
):
    import numpy as np

    unit_column_data_types = {column["name"]: column["data_type"] for column in unit_column_info}

    sorting_extractor = sorting_interface.sorting_extractor

    for entry_index, entry in enumerate(unit_table_json):
        unit_properties = dict(entry)  # copy

        unit_id = unit_properties.pop("unit_id", None)  # NOTE: Is called unit_name in the actual units table

        for property_name, property_value in unit_properties.items():

            if property_name == "unit_id":
                continue  # Already controlling unit_id with the above variable

            dtype = unit_column_data_types[property_name]
            if property_name in SORTING_INTERFACE_PROPERTIES_TO_RECAST:
                property_value = [property_value]
                dtype = "object"  # Should allow the array to go through

            sorting_extractor.set_property(
                key=property_name,
                values=np.array([property_value], dtype=dtype),
                ids=[int(unit_id)],
                # ids=[unit_id]
            )
