import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory

from routes.image_processing import image_bp
from flask_cors import CORS

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config["SECRET_KEY"] = 'adf'


CORS(app)


app.register_blueprint(image_bp, url_prefix='/api/image')




@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    
    
    if path == 'tuition' or path == 'tuition.html':
        tuition_path = os.path.join(static_folder_path, 'tuition.html')
        if os.path.exists(tuition_path):
            return send_from_directory(static_folder_path, 'tuition.html')
        else:
            return "tuition.html not found", 404
    
    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


@app.route('/tuition')
def tuition():
    return send_from_directory(app.static_folder, 'tuition.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
