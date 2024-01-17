from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 仮のTodoリストを作成します
todos = []

# Todoリストの画面を表示します。
@app.route('/')
def index():
    return render_template('index.html', todos=todos)

# 追加されたTodoをTodoリストに加えます。
@app.route('/add', methods=['POST'])
def add_todo():
    todo = request.form.get('todo')
    todos.append(todo)
    return redirect(url_for('index'))

# Todoの編集画面を表示します。
@app.route('/edit/<int:todo_id>')
def edit_todo(todo_id):
    # Todoがあるときは編集画面に、そうではないときはTodoリストへ画面遷移します。
    if 1 <= todo_id <= len(todos):
        return render_template('edit.html', todo=todos[todo_id - 1], todo_id=todo_id)
    else:
        return redirect(url_for('index'))

# Todoが更新されたときの処理です。
@app.route('/update/<int:todo_id>', methods=['POST'])
def update_todo(todo_id):
    # 変更されたTodoがある場合はTodoリストに追加する。
    if 1 <= todo_id <= len(todos):
        todo = request.form.get('todo')
        todos[todo_id - 1] = todo
    return redirect(url_for('index'))

# Todoが削除されたときの処理です。
@app.route('/delete/<int:todo_id>')
def delete_todo(todo_id):
    # Todoがあるときは削除する。
    if 1 <= todo_id <= len(todos):
        del todos[todo_id - 1]
    return redirect(url_for('index'))

# アプリを実行する処理です。
if __name__ == '__main__':
    app.run(debug=True)