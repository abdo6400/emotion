from fastapi import FastAPI, File, UploadFile
from typing import Optional
import numpy as np
import librosa
import pandas as pd
import speech_recognition as sr
import uvicorn
import tensorflow as tf
import joblib



loaded_model = tf.keras.models.load_model("model.h5")
sc = joblib.load('scaler1.bin')
encoder = joblib.load('encoder.bin')

app = FastAPI()
r = sr.Recognizer()
path = "file1.wav"


def Feature_Extraction(X, sample_rate):
  result = np.array([])
  stft = np.abs(librosa.stft(X))
  mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
                 axis=0)
  result = np.hstack((result, mfcc))
  mfcc_std = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
                    axis=0)
  result = np.hstack((result, mfcc_std))
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                   axis=0)
  result = np.hstack((result, chroma))
  chroma_std = np.std(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                      axis=0)
  result = np.hstack((result, chroma_std))
  mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
  result = np.hstack((result, mel))
  mel_std = np.std(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,
                   axis=0)
  result = np.hstack((result, mel_std))
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft,
                                                       sr=sample_rate).T,
                     axis=0)
  result = np.hstack((result, contrast))
  contrast_std = np.std(librosa.feature.spectral_contrast(S=stft,
                                                          sr=sample_rate).T,
                        axis=0)
  result = np.hstack((result, contrast_std))
  return result


def get_emo(audio):
  audio, s = librosa.load(audio)
  librosa_X = []
  yt, _ = librosa.effects.trim(audio)
  res1 = Feature_Extraction(yt, s)
  librosa_X.append(res1)
  df_feature = pd.DataFrame(librosa_X)
  scaled = sc.transform(pd.DataFrame(df_feature))
  X_exp = np.expand_dims(scaled, axis=2)
  y = loaded_model.predict(X_exp)
  result = encoder.inverse_transform(y)
  return result


def convert_to_text(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            text = r.recognize_google(audio, language='ar-egy')
            return text
        except sr.RequestError as e:
            return ("Could not request results from Google Speech Recognition service; {0}"
                    .format(e))

@app.post("/api/audio")
async def create_page(messageFile: Optional[UploadFile] = File(...)):
    if messageFile:
        # Save the file
        with open(messageFile.filename, "wb") as buffer:
            buffer.write(messageFile.file.read())

        # Process the file
        emotion = get_emo(messageFile.filename)[0][0]

        # Return the result
        return {
            "emotion": emotion,     
        }

    # If file was not included in the request, return an error response
    return {"response": "file not found"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000)
