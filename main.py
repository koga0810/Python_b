import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classes = ["猫","犬"]
image_size = 150#画像のサイズを１５０に

UPLOAD_FOLDER = "C:\\Users\\6d09\\Desktop\\dog or cat\\uploads"#アップロードされた画像の保存
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])#アップロードを許可する拡張子の指定

app = Flask(__name__)#flaskのオブジェクト作成
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):#'.'がファイル名に含まれ、許可した拡張子を使用している場合にTrueを返す
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# モデルの読み込みをエラーハンドリング
try:
    model = load_model('./model.h5')#学習済みのモデルを使用
    #model_path = os.path.abspath("C:/CompSche/PythonRun/Python_b/model.h5")
    #model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    # ここでエラーが発生した場合、適切な対処を行う
    
# UPLOAD_FOLDERが存在しない場合は作成
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER']) 




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':#httpのメソッドで画像を送信されたときの動作
        
        if 'file' not in request.files:#ファイルがない場合はファイルがありませんの表記
            ans = 'ファイルがありません'
            return render_template("index.html",answer=ans)
        
        file = request.files['file']
        
        if file.filename == '':#ファイル名がない場合はファイル名がありませんの表記
            ans = 'ファイル名がありません'
            return render_template("index.html",answer=ans)
        
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
        except FileNotFoundError:
            print(f"Error: Directory not found - {app.config['UPLOAD_FOLDER']}")

        img = image.load_img(file_path, target_size=(image_size, image_size))#（画像のURL、縦横のサイズ
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)#新しい配列の追加
        data = img / 255.0

        try:
            result = model.predict(data)#detaから予測しresultに入れる
            probability = result[0, 0]
            #result= "{}%".format(math.floor(probability * 100))
            
           
                
            if result > 0.9:#resultが0.9より大きい場合は犬になる
                pred_answer = f"{probability * 100:.2f}%の確率で完全に犬"    
            
            elif result > 0.8:#resultが0.8より大きい場合は犬になる
                pred_answer = f"{probability * 100:.2f}%の確率でほぼ犬"
            
            elif result > 0.5:#resultが0.5より大きい場合は犬になる
                pred_answer = f"{probability * 100:.2f}%の確率でたぶん犬"
                
            elif result > 0.4:#resultが0.4より大きい場合は猫になる
                cat_probability = 1-probability  # 猫の場合の確率を計算
                pred_answer = f"{cat_probability * 100:.2f}%の確率でたぶん猫"    
            
            elif result > 0.3:#resultが0.3より大きい場合は猫になる
                cat_probability = 1-probability  # 猫の場合の確率を計算
                pred_answer = f"{cat_probability * 100:.2f}%の確率でほぼ猫"
            
            elif result > 0.1:#resultが0.1より大きい場合は猫になる
                cat_probability = 1-probability  # 猫の場合の確率を計算
                pred_answer = f"{cat_probability * 100:.2f}%の確率でほぼ確実に猫"
                    
            else:#それ以外
                cat_probability = 1-probability  # 猫の場合の確率を計算
                pred_answer = f"{cat_probability * 100:.2f}%の確率で完全に猫"
                
            pred_answer = f"これは{pred_answer} "#猫か犬か判別する
      
            
        except Exception as e:
            print(f"Error predicting: {e}")
            pred_answer = "予測エラーが発生しました"
            
        try:
            os.remove(file_path)
        except Exception as e:
            pass  # エラーが発生した場合は何もしない   

        return render_template("result.html", answer=pred_answer)  
      
    return render_template("index.html",answer="判定受け付け待ち")



if __name__ == "__main__":
    app.run()   
