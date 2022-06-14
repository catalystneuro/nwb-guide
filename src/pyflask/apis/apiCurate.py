from flask_restx import Resource, fields, reqparse
from flask_restx.inputs import boolean
from namespaces import NamespaceEnum, get_namespace

from curate import (
    create_folder_level_manifest,
    check_empty_files_folders,
    main_curate_function,
    main_curate_function_progress,
    generate_manifest_file_locally,
    check_JSON_size,
    main_curate_function_upload_details,
    create_high_level_manifest_files_existing_local_starting_point,
)
from errorHandlers.notBadRequestException import notBadRequestException

api = get_namespace(NamespaceEnum.CURATE_DATASETS)

model_check_empty_files_folders_response = api.model( "CheckEmptyFilesFoldersResponse", {
    "empty_files": fields.List(fields.String),
    "empty_folders": fields.List(fields.String),
    "soda_json_structure": fields.String(description="JSON structure of the SODA dataset"),
})

@api.route("/empty_files_and_folders")
class CheckEmptyFilesFolders(Resource):
    # response types/codes
    @api.doc(responses={500: 'There was an internal server error', 400: 'Bad Request'}, description="Given a sodajsonobject return a list of empty files and folders should they exist, as well as the sodajsonobject.")
    def get(self):

        # get the soda_json_structure from the request object
        args = reqparse.RequestParser()
        args.add_argument("soda_json_structure", type=dict, required=True, help="SODA's Dataset structure", location="json")

        # get the dataset_structure from the request object
        soda_json_structure = args.parse_args().get("soda_json_structure")

        try:
            return check_empty_files_folders(soda_json_structure)
        except Exception as e:
            api.abort(500, str(e))





model_main_curation_function_response = api.model( "MainCurationFunctionResponse", {
    "main_curate_progress_message": fields.String(description="Progress message from the main curation function"),
    "main_total_generate_dataset_size": fields.String(description="Total size of the dataset"),
    "main_curation_uploaded_files": fields.Integer(description="Number of files that are being generated. ")
})

@api.route("/curation")
class Curation(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument("soda_json_object", type=dict, required=True, help="SODA's Dataset structure", location="json")

    @api.doc(responses={500: 'There was an internal server error', 400: 'Bad Request', 403: 'Forbidden'}, description="Given a sodajsonobject generate a dataset. Used in the final step of Organize Datasets.")
    @api.marshal_with(model_main_curation_function_response)
    @api.expect(parser)
    def post(self):
        # get the soda_json_structure from the request object
        soda_json_object = self.parser.parse_args().get("soda_json_object")

        try:
            return main_curate_function(soda_json_object)
        except Exception as e:
            if notBadRequestException(e):
                api.abort(500, str(e))
            raise e






model_curation_progress_response = api.model( "CurationProgressResponse", {
    "main_curate_status": fields.String(description="Status of the main curation function"),
    "start_generate": fields.Boolean(description="True if the main curation function is running"),
    "main_curate_progress_message": fields.String(description="Progress message from the main curation function"),
    "main_total_generate_dataset_size": fields.Integer(description="Total size of the dataset"),
    "main_generated_dataset_size": fields.Integer(description="Size of the dataset that has been generated thus far"),
    "elapsed_time_formatted": fields.String(description="Elapsed time of the main curation function"),
})

@api.route("curation/progress")
class CurationProgress(Resource):

    @api.marshal_with(model_curation_progress_response, False, 200)
    @api.doc(responses={500: 'There was an internal server error'}, description="Return important details to the client about the state of the currently running curation function.")
    def get(self):
        try:
            return main_curate_function_progress()
        except Exception as e:
            api.abort(500, str(e))








model_curation_file_details_response = api.model( "CurationFileDetailsResponse", {
    "main_curation_uploaded_files": fields.Integer(description="Number of files that have been uploaded thus far. "),
    "current_size_of_uploaded_files": fields.Integer(description="Size of the files that have been uploaded thus far. "),
    "uploaded_folder_counter": fields.Integer(description="Number of folders that have been uploaded thus far. "),
    "generated_dataset_id": fields.String(description="ID of the dataset that has been generated. ")
})

@api.route("curation/upload_details")
class CurationFileDetails(Resource):
    
        @api.marshal_with(model_curation_file_details_response, False, 200)
        @api.doc(responses={500: 'There was an internal server error'}, description="Function frequently called by front end to help keep track of the amount of files that have been successfully uploaded to Pennsieve, and the size of the uploaded files. Also tells us how many files have been copied (double usage of both variables) to a destination folder for local dataset generation.")
        def get(self):
            try:
                return main_curate_function_upload_details()
            except Exception as e:
                api.abort(500, str(e))







model_generate_manifest_files_response = api.model( "GenerateManifestFilesResponse", {
    "soda_json_object": fields.String(description="JSON structure of the SODA dataset"),
})

@api.route("/manifest_files/local")
class GenerateManifestFiles(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("filepath", type=dict, required=True, help="Path to either a local dataset or a SODA directory for storing temporary manifest files. The latter is used for editing manifest files for local datasets. The former for editing manifest files of manifest information pulled from Pennsieve.", location="json")
    

    @api.doc(responses={500: 'There was an internal server error', 400: 'Bad Request'}, description="Generate manifest files in a local temporary directory or in the user's dataset directory. Allows users to edit their manifest files in the standalone manifest file generator feature.")
    @api.marshal_with(model_generate_manifest_files_response)
    @api.expect(parser)
    def post(self):   # sourcery skip: use-named-expression

        # get the filepath from the request object
        data = self.parser.parse_args()

        filepath = data.get("filepath")

        try:
            return create_high_level_manifest_files_existing_local_starting_point(filepath)
        except Exception as e:
            api.abort(500, str(e))