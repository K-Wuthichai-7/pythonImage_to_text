# from flask import Flask, render_template, request, redirect, url_for
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)

#     file = request.files['file']

#     if file.filename == '':
#         return redirect(request.url)

#     if file:
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filename)
#         return 'ไฟล์ถูกอัปโหลดเรียบร้อย!'

# if __name__ == '__main__':
#     app.run(debug=True)
