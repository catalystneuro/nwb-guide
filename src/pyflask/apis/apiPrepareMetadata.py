from prepareMetadata import (
    save_submission_file,
    save_ds_description_file,
    extract_milestone_info,
    import_milestone,
    save_subjects_file,
    convert_subjects_samples_file_to_df,
    save_samples_file,
    load_taxonomy_species,
    load_existing_DD_file,
    load_existing_submission_file,
    import_bf_metadata_file,
    import_bf_RC,
    upload_RC_file,
    delete_manifest_dummy_folders,
    set_template_path
)
from namespaces import NamespaceEnum, get_namespace
from flask_restx import Resource, reqparse, fields
from errorHandlers import notBadRequestException

api = get_namespace(NamespaceEnum.PREPARE_METADATA)




model_save_submission_file_response = api.model('saveSubmissionFileResponse', {
    'size': fields.Integer(required=True, description='Size of the file in bytes'),
})

model_get_submission_file_response = api.model('getSubmissionFileResponse', {
    "SPARC Award number": fields.String(required=True, description='SPARC Award number'),
    "Milestone achieved": fields.List(fields.String, required=True, description='Milestone achieved'),
    "Milestone completion date": fields.String(required=True, description='Milestone completion date'),
})
@api.route('/submission_file')
class SaveSubmissionFile(Resource):

    parser_save_submission_file = reqparse.RequestParser(bundle_errors=True)
    parser_save_submission_file.add_argument('upload_boolean', type=bool, help='Boolean to indicate whether to upload the file to the Bionimbus server', location="json", required=True)
    parser_save_submission_file.add_argument('selected_account', type=str, help='Pennsieve account name', location="args", required=True)
    parser_save_submission_file.add_argument('selected_dataset', type=str, help='Pennsieve dataset name', location="args", required=True)
    parser_save_submission_file.add_argument('filepath', type=str, help='Path to the file to be uploaded', location="json")
    parser_save_submission_file.add_argument('json_str', type=str, help='JSON string to be uploaded', location="json", required=True)

    @api.expect(parser_save_submission_file)
    @api.response(200, 'OK', model_save_submission_file_response)
    @api.doc(description='Save a submission file locally or in a dataset stored on the Pennsieve account of the current user.', responses={500: "Internal Server Error", 400: "Bad Request"})
    def post(self):
        data = self.parser_save_submission_file.parse_args()

        upload_boolean = data.get('upload_boolean')
        bfaccount = data.get('selected_account')
        bfdataset = data.get('selected_dataset')
        filepath = data.get('filepath')
        json_str = data.get("json_str")


        if upload_boolean and filepath is None:
            api.abort(400, "Error: Please provide a destination in which to save your Submission file.")

        try:
            return save_submission_file(upload_boolean, bfaccount, bfdataset, filepath, json_str)
        except Exception as e:
            api.abort(500, str(e))



    parser_get_submission_file = reqparse.RequestParser(bundle_errors=True)
    parser_get_submission_file.add_argument('filepath', type=str, help="Path to the submission file on the user's machine", location="json", required=True)


    @api.expect(parser_get_submission_file)
    @api.marshal_with(model_get_submission_file_response, 200, False)
    @api.doc(description='Get the submission file from the user\'s machine.', responses={500: "Internal Server Error", 400: "Bad Request"})
    def get(self):
        data = self.parser_get_submission_file.parse_args()
        filepath = data.get('filepath')

        try:
            return load_existing_submission_file(filepath)
        except Exception as e:
            if notBadRequestException(e):
                api.abort(500, str(e))
            raise e




@api.route('/template_paths')
class SetTemplatePath(Resource):

    parser_set_template_path = reqparse.RequestParser(bundle_errors=True)
    parser_set_template_path.add_argument('basepath', type=str, help='Path to the template directory', location="json")
    parser_set_template_path.add_argument('resourcesPath', type=str, help='Path to the template directory', location="json")

    @api.expect(parser_set_template_path)
    def put(self):
        data = self.parser_set_template_path.parse_args()
        basepath = data.get('basepath')
        resourcesPath = data.get('resourcesPath')

        if basepath is None and resourcesPath is None:
            return api.abort(400, "Missing required parameters: basepath, resourcesPath")

        try:
            return set_template_path(basepath, resourcesPath)
        except Exception as e:
            api.abort(500, str(e))







parser_import_milestone = reqparse.RequestParser(bundle_errors=True)
parser_import_milestone.add_argument('path', type=str, help='Path to the local data deliverables document', location="args")
@api.route('/import_milestone')
class ImportMilestone(Resource):

    @api.expect(parser_import_milestone)
    def get(self):
        args = parser_import_milestone.parse_args()
        path = args['path']
        try:
            return import_milestone(path)
        except Exception as e:
            if notBadRequestException(e):
                api.abort(500, e.args[0])
            raise e