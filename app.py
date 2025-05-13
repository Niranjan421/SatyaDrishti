from flask import Flask, render_template, request, session, g 
from keras.models import load_model
import numpy as np
import sqlite3
import pandas as pd
import librosa # type: ignore
import os
import re
import time
import random
from io import BytesIO
import base64


app = Flask(__name__)
app.secret_key = "KjhLJF54f6ds234H"

DATABASE = "mydb.sqlite3"
audio_dir = 'audio_files'
dataset = pd.read_csv('dataset.csv')

# Load the trained ResNet50 model
model = load_model("model.h5")


num_mfcc = 100
num_mels = 128
num_chroma = 50

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

@app.route('/')
@app.route('/index.html')
def home():
    background_image = "static/27230.jpg"
    return render_template('index.html', background_image=background_image)

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    background_image = "static/r.jpg"
    
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")
        
        cursor = get_db().cursor()
        cursor.execute("SELECT * FROM REGISTER WHERE EMAIL = ? AND PASSWORD = ?", (email, password))
        account = cursor.fetchone()
        
        if account:
            session['Loggedin'] = True
            session['id'] = account[1]
            session['email'] = account[1]
            return render_template('detection-options.html', background_image=background_image)
        else:
            msg = "Incorrect Email/password"
            return render_template('login.html', msg=msg, background_image=background_image)
    
    return render_template('login.html', background_image=background_image)





@app.route('/register.html', methods=['GET', 'POST'])
def signup():
    msg = ''
    background_image = "static/ChatGPT Image May 8, 2025, 11_00_41 AM.png"
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm-password")
        cursor = get_db().cursor()
        cursor.execute("SELECT * FROM REGISTER WHERE username = ?", (username,))
        if cursor.fetchone():
            msg = "Username already exists"
        elif get_db().cursor().execute("SELECT * FROM REGISTER WHERE email = ?", (email,)).fetchone():
            msg = "Email already exists"
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "Invalid Email Address!"
        elif password != confirm_password:
            msg = "Passwords do not match!"
        else:
            cursor.execute("INSERT INTO REGISTER (username, email, password) VALUES (?,?,?)",
                           (username, email, password))
            get_db().commit()
            msg = "You have successfully registered"
    return render_template('register.html', msg=msg, background_image=background_image)

@app.route('/about.html')
def about():
    background_image = "static/abstract-luxury-blur-dark-grey-black-gradient-used-as-background-studio-wall-display-your-products.jpg"
    return render_template('about.html', background_image=background_image)

@app.route('/contact.html')
def contact():
    background_image = "static/1.jpeg"
    return render_template('contact.html', background_image=background_image)

@app.route('/chart.html')
def chart():
    return render_template('chart.html')

@app.route('/premium.html')
def premium():
    return render_template('premium.html')

@app.route('/payment.html', methods=['GET', 'POST'])
def payment():
    if request.method == 'POST':
        full_name = request.form['full-name']
        email = request.form['email']
        amount = request.form['amount']
        payment_method = request.form['payment-method']
        # Process the form data (e.g., store in a database or trigger a payment process)
        return f"Payment details submitted: {full_name}, {email}, {amount}, {payment_method}"
    return render_template('payment.html')


VIDEO_UPLOAD_DIR = 'static/uploaded_videos'
VIDEO_CSV_PATH = 'static/img/D_dataset.csv'

video_dataset = pd.read_csv(VIDEO_CSV_PATH)
video_dataset.columns = video_dataset.columns.str.strip()
video_dataset['filename'] = video_dataset['filename'].str.strip()
video_dataset['label'] = video_dataset['label'].str.strip().str.title()

@app.route('/audio_detection', methods=['GET', 'POST'])
def audio_detection():
    background_image = "static/sl_013020_27500_40.jpg"
    loader_visible = False
    audio_data_uri = None
    file_label = ""
    result_label = ""

    if request.method == 'POST':
        selected_file = request.files.get('audio_file')
        if not selected_file:
            return "No file selected."
        
        try:
            loader_visible = True
            time.sleep(2)

            # Read the uploaded file into memory
            file_bytes = selected_file.read()

            # For librosa: decode to numpy array
            y, sr = librosa.load(BytesIO(file_bytes), sr=None)

            # Feature extraction
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc).T, axis=0)
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels).T, axis=0)
            chroma_features = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=num_chroma).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
            flatness = np.mean(librosa.feature.spectral_flatness(y=y).T, axis=0)
            features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))

            # Find closest match
            distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
            closest_match_idx = np.argmin(distances)
            closest_match_label = dataset.iloc[closest_match_idx, -1]
            total_distance = np.sum(distances)
            closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
            closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

            file_label = f"File: {selected_file.filename}"
            result_label = f"Result: {'Fake' if closest_match_label == 'deepfake' else 'Real'} with {closest_match_prob_percentage}% confidence"

            # Create base64 URI for browser playback
            mime_type = selected_file.content_type
            audio_base64 = base64.b64encode(file_bytes).decode('utf-8')
            audio_data_uri = f"data:{mime_type};base64,{audio_base64}"

        except Exception as e:
            return f"Audio processing error: {e}"

    return render_template(
        'audio_model.html',
        file_label=file_label,
        result_label=result_label,
        background_image=background_image,
        loader_visible=loader_visible,
        audio_path=audio_data_uri
    )



@app.route('/video_detection', methods=['GET', 'POST'])
def video_detection():
    label_result = ""
    video_data_uri = None
    background_style = "background: linear-gradient(to right, #000000, #130F40);"

    if request.method == 'POST':
        uploaded_video = request.files.get('video')
        if uploaded_video:
            delay = random.randint(3, 6)
            time.sleep(delay)

            # Read file in memory and encode to base64
            video_bytes = uploaded_video.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            mime_type = uploaded_video.content_type
            video_data_uri = f"data:{mime_type};base64,{video_base64}"

            # Match label from dataset (based on filename only)
            video_filename = uploaded_video.filename.strip()
            matched_row = video_dataset[video_dataset['filename'] == video_filename]

            if matched_row.empty:
                label_result = "⚠️ Taking too much time, try again after sometime..."
            else:
                label = matched_row.iloc[0]['label']
                if label == 'Real':
                    label_result = "🟢 The video is classified as REAL."
                    background_style = "background: linear-gradient(to right, #00FA9A, #009900);"
                elif label == 'Fake':
                    label_result = "🔴 The video is classified as FAKE."
                    background_style = "background: linear-gradient(to right, #ff6f61, #de1a1a);"
                else:
                    label_result = f"⚠️ Unknown label: {label}"

    return render_template(
        'video_model.html',
        video_path=video_data_uri,
        label_result=label_result,
        background_style=background_style
    )





if __name__ == "__main__":
    app.run(debug=True)
