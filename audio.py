from flask import Flask, render_template, request, session, g
import numpy as np
import sqlite3
import re
import pandas as pd
import librosa
import os
import time

app = Flask(__name__)
app.secret_key = "KjhLJF54f6ds234H"

DATABASE = "mydb.sqlite3"
audio_dir = 'audio_files'
dataset = pd.read_csv('dataset.csv')

num_mfcc = 100
num_mels = 128
num_chroma = 50

def get_db():
    """Get database connection."""
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection on teardown."""
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    """Render the home page."""
    background_image = "/static/95.png"
    return render_template('index.html', background_image=background_image)

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    """Handle login requests."""
    background_image = "/static/45.jpg"
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
            return render_template('model.html', background_image=background_image)
        else:
            msg = "Incorrect Email/password"
            return render_template('login.html', msg=msg, background_image=background_image)
    
    return render_template('login.html', background_image=background_image)

@app.route('/contact.html')
def contact():
    """Render the contact page."""
    background_image = "/static/98.jpg"
    return render_template('contact.html', background_image=background_image)

@app.route('/about.html')
def about():
    """Render the about page."""
    background_image = "/static/2.jpg"
    return render_template('about.html', background_image=background_image)

@app.route('/index.html')
def home1():
    """Render the home page."""
    background_image = "/static/95.png"
    return render_template('index.html', background_image=background_image)

@app.route('/chart.html')
def chart():
    """Render the chart page."""
    return render_template('chart.html')

@app.route('/register.html', methods=['GET', 'POST'])
def signup():
    """Handle registration requests."""
    msg = ''
    background_image = "/static/60.jpg"
    
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm-password")
        cursor = get_db().cursor()
        cursor.execute("SELECT * FROM REGISTER WHERE username = ?", (username,))
        account_username = cursor.fetchone()
        cursor.execute("SELECT * FROM REGISTER WHERE email = ?", (email,))
        account_email = cursor.fetchone()

        if account_username:
            msg = "Username already exists"
        elif account_email:
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

@app.route('/model.html', methods=['GET', 'POST'])
def model():
    """Handle audio file upload and deepfake detection."""
    background_image = "/static/56.jpg"
    loader_visible = False

    if request.method == 'POST':
        selected_file = request.files.get('audio_file')
        
        if not selected_file:
            return "No file selected. Please upload a valid audio file."
        
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        
        file_path = os.path.join(audio_dir, selected_file.filename)
        try:
            selected_file.save(file_path)
        except Exception as e:
            return f"Error saving the file: {e}"

        loader_visible = True
        time.sleep(2)

        try:
            X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        except Exception as e:
            return f"Error processing the audio file: {e}"

        try:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
            chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
            flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
            features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
        except Exception as e:
            return f"Error extracting features from the audio file: {e}"

        try:
            distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
            closest_match_idx = np.argmin(distances)
            closest_match_label = dataset.iloc[closest_match_idx, -1]
            total_distance = np.sum(distances)
            closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
            closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)
        except Exception as e:
            return f"Error calculating the closest match: {e}"

        file_label = f"File: {os.path.basename(file_path)}"
        result_label = f"Result: {'Fake' if closest_match_label == 'deepfake' else 'Real'} with {closest_match_prob_percentage}% confidence"

        try:
            os.remove(file_path)
        except Exception as e:
            return f"Error deleting the file: {e}"

        return render_template('model.html', file_label=file_label, result_label=result_label, background_image=background_image, loader_visible=loader_visible)
    
    return render_template('model.html', background_image=background_image, loader_visible=loader_visible)

if __name__ == "__main__":
    app.run(debug=True)
