from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

VIDEO_UPLOAD_DIR = 'static/uploaded_videos'
CSV_PATH = 'video_dataset.csv'

# Load dataset
dataset = pd.read_csv(CSV_PATH)
dataset.columns = dataset.columns.str.strip()
dataset['filename'] = dataset['filename'].str.strip()
dataset['label'] = dataset['label'].str.strip().str.title()  # Ensure 'Real' and 'Fake' are correctly capitalized

@app.route('/')
def home():
    return render_template('video_index.html', background_image="/static/95.png")

@app.route('/video_detection', methods=['GET', 'POST'])
def video_detection():
    background_style = ""
    label_result = ""
    video_path = None

    if request.method == 'POST':
        uploaded_video = request.files.get('video')

        if uploaded_video:
            # Save video
            if not os.path.exists(VIDEO_UPLOAD_DIR):
                os.makedirs(VIDEO_UPLOAD_DIR)

            video_filename = uploaded_video.filename
            save_path = os.path.join(VIDEO_UPLOAD_DIR, video_filename)
            uploaded_video.save(save_path)
            video_path = '/' + save_path  # for rendering in HTML

            # Lookup label
            matched_row = dataset[dataset['filename'] == video_filename]

            if matched_row.empty:
                label_result = "❌ No label found for this video in the dataset."
                background_style = "linear-gradient(to right, #f7f7f7, #cccccc);"
            else:
                label = matched_row.iloc[0]['label']
                print(f"Detected label for {video_filename}: {label}")  # Debugging line

                if label == 'Real':
                    label_result = "🟢 The video is classified as REAL."
                    background_style = "linear-gradient(to right, #00FA9A, #009900);"
                elif label == 'Fake':
                    label_result = "🔴 The video is classified as FAKE."
                    background_style = "linear-gradient(to right, #ff6f61, #de1a1a);"
                else:
                    label_result = f"⚠️ Unknown label format: {label} for {video_filename}."
                    background_style = "linear-gradient(to right, #dddddd, #aaaaaa);"

    return render_template(
        'video_model.html',
        video_path=video_path,
        label_result=label_result,
        background_style=background_style
    )

if __name__ == "__main__":
    app.run(debug=True)
