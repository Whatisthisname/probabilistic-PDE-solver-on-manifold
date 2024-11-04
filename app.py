from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route("/json-files", methods=["GET"])
def list_json_files():
    try:
        files = [
            f
            for f in os.listdir(os.path.join(".", "computed_meshes"))
            if f.endswith(".json")
        ]
        files.sort()
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/<path:filename>", methods=["GET"])
def serve_file(filename):
    print("hey:", filename)
    try:
        return send_from_directory("./computed_meshes", filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
