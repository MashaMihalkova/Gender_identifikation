import wave
import numpy as np
import os
from keras.models import load_model
import optparse
import logging
from logging import error, info
import librosa

def wav2mfcc(path, max_pad_size=11):
  y, sr = librosa.load(path=path, sr=None, mono=1)
  y = y[::3]
  audio_mac = librosa.feature.mfcc(y=y, sr=8000)
  y_shape = audio_mac.shape[1]
  if y_shape < max_pad_size:
      pad_size = max_pad_size - y_shape
      audio_mac = np.pad(audio_mac, ((0, 0), (0, pad_size)), mode='constant')
  else:
      audio_mac = audio_mac[:, :max_pad_size]
  return audio_mac




if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option('-m', '--model', type=str,
          help="path to the model", default="/content/drive/MyDrive/iden_gender/my_model1.h5")
    parser.add_option('-f', '--file', type=str,
          help=" path to the file for testing", default="/content/drive/MyDrive/iden_gender/inner_test/Sound_22123.wav")

    options, args = parser.parse_args()     
    MODEL   = getattr(options,'model')
    FILE    = getattr(options,'file') 

    # if not os.path.isfile(MODEL) or not os.stat(MODEL).st_size:
    #   error("Weights not found. Unable to load the model.")
    
    # if not os.path.isfile(FILE):
    #   error("File not found. Unable to load file.")
      
    try:
      model = load_model(MODEL) 
        
      info(f'Loaded model: {MODEL}.')

      wavs=[]
      wavs.append(wav2mfcc(FILE,11))
      X=np.array(wavs)
      X= X.reshape(-1, 220)
      
      result=model.predict(X[0:1])[0] # 
      print("Prediction result",result)
     
      name = ["male","female"]
      ind=0 
      for i in range(len(result)):
          if result[i] > result[ind]:
              ind=1
      print("Gender:",name[ind])
    except Exception as e:
      error(str(e))

