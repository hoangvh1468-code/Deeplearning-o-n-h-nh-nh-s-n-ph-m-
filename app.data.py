import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from product_recognition import ProductRecognitionSystem

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.secret_key = 'replace-this-secret'

system = ProductRecognitionSystem(
    data_dir=os.path.join(BASE_DIR, 'data', 'raw'),
    model_dir=os.path.join(BASE_DIR, 'models')
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

system.prepare_data()
if os.path.exists(system.model_path) and os.path.exists(system.encoder_path):
    system.load_model()
else:
    system.build_model()
    system.train()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='Vui lòng chọn ảnh để upload.', result=None)

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='Vui lòng chọn ảnh hợp lệ.', result=None)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(saved_path)

            result = system.predict(saved_path)
            image_url = url_for('uploaded_file', filename=filename)
            return render_template('result.html', result=result, image_url=image_url)

        return render_template('index.html', error='Định dạng ảnh không hỗ trợ.', result=None)

    return render_template('index.html', error=None, result=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
