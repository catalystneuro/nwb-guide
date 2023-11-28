"""API endpoint definitions for interacting with NeuroConv."""
import traceback

from flask_restx import Namespace, Resource, reqparse

from manageNeuroconv import generate_tutorial_data
from errorHandlers import notBadRequestException

tutorial_api = Namespace("tutorial", description="API route for tutorial operations in the NWB GUIDE.")


@tutorial_api.errorhandler(Exception)
def exception_handler(error):
    exceptiondata = traceback.format_exception(type(error), error, error.__traceback__)
    return {"message": exceptiondata[-1], "traceback": "".join(exceptiondata)}


# @tutorial_api.route("/generate/<string:base_path>")
# class GenerateTutorialData(Resource):
#     @tutorial_api.doc(responses={200: "Success", 400: "Bad Request", 500: "Internal server error"})
#     def post(self, base_path: str):
#         try:
#             generate_tutorial_data(base_path=base_path)
#         except Exception as exception:
#             if notBadRequestException(exception):
#                 tutorial_api.abort(500, str(exception))
#             raise exception

generate_tutorial_data_parser = reqparse.RequestParser()
generate_tutorial_data_parser.add_argument("base_path", type=str)


@tutorial_api.route("/generate")
@tutorial_api.expect(generate_tutorial_data_parser)
class GenerateTutorialData(Resource):
    @tutorial_api.doc(responses={200: "Success", 400: "Bad Request", 500: "Internal server error"})
    def post(self):
        try:
            generate_tutorial_data(**tutorial_api.payload)
        except Exception as exception:
            if notBadRequestException(exception):
                tutorial_api.abort(500, str(exception))
            raise exception
