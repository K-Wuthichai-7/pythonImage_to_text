from flask import Flask, render_template, request, redirect, url_for
import os
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from pytesseract import Output
from tabulate import tabulate
import pandas as pd
import numpy as np
import pytesseract
import imutils
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to display images
def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

# Function to perform OCR and image processing
def perform_ocr(image_path, args):
    # Your OCR and Image Processing code...
    np.random.seed(42)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ...

    return image, gray

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        args = {
            "image": filename,  # เปลี่ยนค่า image เป็นตัวแปร filename
            "output": "results.csv",
            "min_conf": 0,
            "dist_thresh": 25.0,
            "min_size": 2,
        }

        np.random.seed(42)
        image = cv2.imread(args["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 11))
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        grad = (grad - minVal) / (maxVal - minVal)
        grad = (grad * 255).astype("uint8")

        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        tableCnt = max(cnts, key=cv2.contourArea)

        options = "--psm 11"
        results = pytesseract.image_to_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), config=options, output_type=Output.DICT)

        coords = []
        ocrText = []

        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]

            text = results["text"][i]
            conf = int(float(results["conf"][i]))

            if conf > args["min_conf"]:
                coords.append((x, y, w, h))
                ocrText.append(text)

        xCoords = [(c[0], 0) for c in coords]

        clustering = AgglomerativeClustering(n_clusters=None, linkage="complete", distance_threshold=args["dist_thresh"])
        clustering.fit(xCoords)

        sortedClusters = []

        for l in np.unique(clustering.labels_):
            idxs = np.where(clustering.labels_ == l)[0]

            if len(idxs) > args["min_size"]:
                avg = np.average([coords[i][0] for i in idxs])
                sortedClusters.append((l, avg))

        sortedClusters.sort(key=lambda x: x[1])
        df = pd.DataFrame()

        for (l, _) in sortedClusters:
            idxs = np.where(clustering.labels_ == l)[0]
            yCoords = [coords[i][1] for i in idxs]
            sortedIdxs = idxs[np.argsort(yCoords)]

            color = np.random.randint(0, 255, size=(3,), dtype="int")
            color = [int(c) for c in color]

            for i in sortedIdxs:
                (x, y, w, h) = coords[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            cols = [ocrText[i].strip() for i in sortedIdxs]
            currentDF = pd.DataFrame({cols[0]: cols[1:]})
            df = pd.concat([df, currentDF], axis=1)

        df.fillna("", inplace=True)
        print(tabulate(df, headers="keys", tablefmt="psql"))

        # Perform OCR and image processing
        image, gray = perform_ocr(filename, args)

        # Your existing code for OCR and clustering
        # ...

        # Print the tabulated result
        df.fillna("", inplace=True)
        result_table = tabulate(df[['PROPERTY', 'RESULT']], headers="keys", tablefmt="psql")
        print(result_table)
        
		# Convert DataFrame to JSON
        json_result = df[['PROPERTY', 'RESULT']].to_dict(orient='records')

        # # Display the JSON result
        # print(json_result)

        return render_template('index.html', image_path=filename, json_result=json_result)

if __name__ == '__main__':
    app.run(debug=True)
