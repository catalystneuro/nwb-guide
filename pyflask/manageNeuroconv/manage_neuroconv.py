from typing import List, Dict
from neuroconv.datainterfaces import SpikeGLXRecordingInterface, PhySortingInterface
from neuroconv import datainterfaces, NWBConverter

import json
from neuroconv.utils import NWBMetaDataEncoder
import nwbinspector
from pynwb.file import NWBFile, Subject
from nwbinspector.nwbinspector import InspectorOutputJSONEncoder

from pathlib import Path
import os
from datetime import datetime


def get_all_interface_info() -> dict:
    """Format an information structure to be used for selecting interfaces based on modality and technique."""
    # Hard coded for now - eventual goal will be to import this from NeuroConv
    interfaces_by_modality_and_technique = dict(
        ecephys=dict(
            recording=dict(SpikeGLX=SpikeGLXRecordingInterface),
            sorting=dict(Phy=PhySortingInterface),
        )
    )

    interface_info = dict()

    for modality, techniques in interfaces_by_modality_and_technique.items():
        for technique, format_name_to_interface in techniques.items():
            for format_name, interface in format_name_to_interface.items():
                # interface = format_name_to_interface
                interface_info[format_name] = {  # Note in the full scope, format_name won't be unique
                    "modality": modality,
                    "name": interface.__name__,  # Where is this value used in the display?
                    "technique": technique,  # Is this actually necessary anymore?
                }

    return interface_info


# Combine Multiple Interfaces
def get_custom_converter(interface_class_names: List[str]) -> NWBConverter:
    class CustomNWBConverter(NWBConverter):
        data_interface_classes = {interface: getattr(datainterfaces, interface) for interface in interface_class_names}

    return CustomNWBConverter


def instantiate_custom_converter(source_data: Dict[str, str]) -> NWBConverter:
    interface_class_names = list(
        source_data
    )  # NOTE: We currently assume that the keys of the properties dict are the interface names
    CustomNWBConverter = get_custom_converter(interface_class_names)
    return CustomNWBConverter(source_data)


def get_source_schema(interface_class_names: List[str]) -> dict:
    """
    Function used to get schema from a CustomNWBConverter that can handle multiple interfaces
    """
    CustomNWBConverter = get_custom_converter(interface_class_names)
    return CustomNWBConverter.get_source_schema()


def get_metadata_schema(source_data: Dict[str, dict]) -> Dict[str, dict]:
    """
    Function used to fetch the metadata schema from a CustomNWBConverter instantiated from the source_data.
    """

    converter = instantiate_custom_converter(source_data)
    schema = converter.get_metadata_schema()
    metadata = converter.get_metadata()
    return json.loads(json.dumps(dict(results=metadata, schema=schema), cls=NWBMetaDataEncoder))



class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def validate_metadata(parent: dict, function: str) -> dict:
    """
    Function used to validate data using an arbitrary NWB Inspector function
    """
    fn = nwbinspector.__dict__.get(function)
    if fn is None:
        raise ValueError(f"Function {function} not found in nwbinspector")
    
    if "subject" in function and "subject_exists" not in function:
        cls = Subject
        default_parent = {}
    else:
        cls = NWBFile
        default_parent = dict(
            session_description="This is a description",
            identifier="00001",
            session_start_time=datetime.now(),
        )

    result = fn(cls(
        **default_parent,
        **parent
    ))

    return json.loads(json.dumps(result, cls=InspectorOutputJSONEncoder))



def convert_to_nwb(info: dict) -> str:
    """
    Function used to convert the source data to NWB format using the specified metadata.
    """

    nwbfile_path = Path(info["nwbfile_path"])

    run_stub_test = info.get("stub_test")

    # add a subdirectory to a filepath if stub_test is true
    if run_stub_test:
        stub_subfolder = nwbfile_path.parent / ".stubs"
        stub_subfolder.mkdir(exist_ok=True)
        preview_path = stub_subfolder / nwbfile_path.name

    converter = instantiate_custom_converter(info["source_data"])

    # Assume all interfaces have the same conversion options for now
    available_options = converter.get_conversion_options_schema()
    options = (
        {
            interface: {"stub_test": info["stub_test"]}
            if available_options.get("properties").get(interface).get("properties").get("stub_test")
            else {}
            for interface in info["source_data"]
        }
        if run_stub_test
        else None
    )

    file = converter.run_conversion(
        metadata=info["metadata"],
        nwbfile_path=preview_path if run_stub_test else nwbfile_path,
        overwrite=info.get("overwrite", False),
        conversion_options=options,
    )

    return str(file)
