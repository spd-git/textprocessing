from flask import Flask
from flask import jsonify
from flask import request
from pprint import pprint


app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """
    Does Health Check
    """
    response_object = {"success": "1","project":"ritiksparser"}
    response = jsonify(response_object)
    response.status_code = 200
    print('Health OK')
    return response

@app.route('/', methods=['GET'])
def index():
    """
    Does Health Check
    """
    response_object = {"Hello":"World", "success": "1","project":"ritiksparser"}
    response = jsonify(response_object)
    response.status_code = 200
    print('Default OK')
    return response

@app.route('/gettime', methods=['GET'])
def gettime():
    """
    Does Health Check
    """
    response_object = {"Hello":"World", "success": "1","project":"ritiksparser"}
    response = jsonify(response_object)
    response.status_code = 200
    print('Default OK')
    return response

@app.route('/endpoint', methods=['POST'])
def endpoint():
    """
    Does Health Check
    """
    json_object = request.get_json()
    response_object = {"endpoint":"endpoint","project":"ritiksparser"}
    response = jsonify(response_object)
    response.status_code = 200
    print('Endpoint OK')
    pprint(json_object)
    pprint(request)
    return response


if __name__ == '__main__':
    app.run()