import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classes = ["ねこ", "いぬ"]
image_size = 150

app = Flask(__name__)

# アップロードする画像を保存するディレクトリを設定
# UPLOAD_FOLDER = "C:\\work\\Classifier\\uploads"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# サポートするファイルの拡張子を設定
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# モデルの読み込みをエラーハンドリング
try:
    model = load_model('./model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    # ここでエラーが発生した場合、適切な対処を行う

# UPLOAD_FOLDERが存在しない場合は作成
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    file_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            ans = 'ファイルがありません'
            return render_template("index.html", answer=ans, file_path=file_path)

        file = request.files['file']

        if file.filename == '':
            ans = 'ファイルがありません'
            return render_template("index.html", answer=ans, file_path=file_path)
        print("File path:", file_path)
        # ファイルパスを構築
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # ファイルパスのデバッグ用
        print(f"File path: {file_path}")

        # フォルダが存在しない場合は作成
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # ファイル保存
        try:
            file.save(file_path)
            print("File saved at:", file_path)
        except FileNotFoundError:
            print(f"Error: Directory not found - {app.config['UPLOAD_FOLDER']}")

        img = image.load_img(file_path, target_size=(image_size, image_size))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        data = img / 255.0

        try:
            result = model.predict(data)
            result = round(result[0, 0])
            pred_answer = "これは " + classes[result] + " です"
        except Exception as e:
            print(f"Error predicting: {e}")
            pred_answer = "予測エラーが発生しました"

        return render_template("index.html", answer=pred_answer, file_path=file_path)

    return render_template("index.html", answer="判定受け付け待ち")

if __name__ == "__main__":
    app.run(debug=True)
