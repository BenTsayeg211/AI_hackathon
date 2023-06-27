from flask import Flask, render_template, request
from io import TextIOBase
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'gif', 'tiff'}


@app.route('/upload', methods=['POST'])
def upload() -> str:
    file = request.files['file']
    if file and allowed_file(file.filename):
        with file.stream as f:
            f: TextIOBase
            content = f.read()
        data_content = base64.b64encode(content).decode()
        html = '<img src="data:image/jpeg;base64,' + data_content + '">'  # embed in html
        return "<h1>File uploaded successfully</h1></br>" + html
    else:
        return "Invalid file extension"


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
