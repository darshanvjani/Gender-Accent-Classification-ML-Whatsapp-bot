import os
import wget
import requests
from urllib.request import Request,urlopen
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import sklearn
import librosa

import pickle
transformer = pickle.load(open('scaler','rb')) #the one which is trained on whole dataset (librosa + freq)1.32M
gender = pickle.load(open('gender_withoutSMT.pkl','rb')) #the one which is trained on whole dataset (librosa + freq)1.32M
#mf_transformer = pickle.load(open('mf_scaler.pkl','rb'))
transformer_old_data = pickle.load(open('scaler.sav','rb'))
male_female_old_data = pickle.load(open('male_female.sav','rb'))
accent = pickle.load(open('accent.pkl','rb'))  #the one which is trained on whole dataset (librosa + freq)1.32M

import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment


app = Flask(__name__)

@app.route("/sms", methods=['POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Fetch the me

    msg = request.form.get('MediaUrl0')    # Create reply
    #resp = MessagingResponse()
    #resp.message("You said: {}".format(msg))
    
    """req = Request(msg, headers={'User-Agent': 'Mozilla/5.0'})
    #webpage = urlopen(req).read().decode('cp437')
    """
    webpage = requests.get(msg)
    print(webpage.url)
    file_name = str(wget.download(webpage.url))
    
    '''src_filename = file_name
    dest_filename = 'output.wav'

    process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
    if process.returncode != 0:
        raise Exception("Something went wrong")
    
    type(process)
    '''
    base = os.path.splitext(file_name)[0]
    os.rename(file_name, base + '.ogg')
    path = os.path.dirname(os.path.abspath(__file__))
    ogg_file_name = str(file_name+'.ogg')
    input_path = str(path+'/'+ogg_file_name)
    
    print(input_path)
    wav_file_name = str(file_name+'.wav')
    output_path = str(path+'/'+wav_file_name)
    print(output_path)
    
    ogg_version = AudioSegment.from_ogg(input_path)
    ogg_version.export(output_path, format="wav")
    os.remove(input_path)
    
    #wav_file = wave.open('./'+wav_file_name,'rb')
    #print(type(wav_file))    
    #def get_frequencies(x):

  #extract list of dominant frequencies in sliding windows of duration defined by 'step' for each of the 10 wav files and return an array
    #frequencies_lol = [] #lol: list of lists
    rate, data = wavfile.read(output_path)
    y , sr = librosa.load(output_path)
    os.remove(output_path)

    #get dominating frequencies in sliding windows of 200ms
    step = rate//5 #3200 sampling points every 1/5 sec 
    window_frequencies = []

    for i in range(0,len(data),step):
      ft = np.fft.fft(data[i:i+step]) #fft returns the list N complex numbers
      freqs = np.fft.fftfreq(len(ft)) #fftq tells you the frequencies associated with the coefficients
      imax = np.argmax(np.abs(ft))
      freq = freqs[imax]
      freq_in_hz = abs(freq *rate)
      window_frequencies.append(freq_in_hz)
      filtered_frequencies = [f for f in window_frequencies if 20<f<280 and not 46<f<66] 
      # I see noise at 50Hz and 60Hz
      print(filtered_frequencies)
    if len(filtered_frequencies) >= 1: 
    
      nobs, minmax, mean, variance, skew, kurtosis =  stats.describe(filtered_frequencies)
      #median   = np.median(filtered_frequencies)
      mode     = stats.mode(filtered_frequencies).mode[0]
      std      = np.std(filtered_frequencies)
      low,peak = minmax
      q75,q25  = np.percentile(filtered_frequencies, [75 ,25])
      iqr      = q75 - q25
    
      rms = librosa.feature.rms(y=y)
      chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
      spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
      spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
      rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
      zcr = librosa.feature.zero_crossing_rate(y)
      mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
      features = np.array([nobs,mean, skew, kurtosis, mode, std, low, peak, q25, q75, iqr,np.mean(chroma_stft),np.mean(mfcc),np.mean(rms),np.mean(rolloff),np.mean(spec_bw),np.mean(spec_cent),np.mean(zcr)]).reshape(1,-1)
      #mf_feature = np.array([np.mean(rms),np.mean(chroma_stft),np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr),np.mean(mfcc)]).reshape(1,-1)
      #print(frequencies)
      mf_features = np.array([nobs,mean,skew, kurtosis, mode, std, low, peak, q25, q75, iqr]).reshape(1,-1)
      
      #features = np.array(frequencies).reshape(-1,1)
      
      print(features)
      print()
      #print(mf_feature)
      transformed_features = transformer.transform(features)
      #mf_transformed_features_old_data = transformer_old_data.transform(mf_features)
      male_female_pred = gender.predict(transformed_features)
      #male_female_pred_old_data = male_female_old_data.predict_proba(mf_transformed_features_old_data)[0]
      
      accent_pred = accent.predict(transformed_features)
      #age_pred = age.predict(transformed_features)
      print('new: ',male_female_pred)
      #print('old: ',male_female_pred_old_data )
      #print()
      print(accent_pred)
      
      mf_ans = str()
      accent_ans = str()
      age_ans = str()
        
      if male_female_pred == 1:
        mf_ans = 'Male'
      else:
        mf_ans = 'Female'
    
      if accent_pred == 0:
        accent_ans = 'American'
      elif accent_pred == 1:
        accent_ans = 'Australian'
      elif accent_pred == 2:
        accent_ans = 'British'
      elif accent_pred == 3:
        accent_ans = 'Europian'
      elif accent_pred == 4:
        accent_ans = 'Indian'
      else:
        accent_ans = 'Other'
      '''  
      if age_pred == 1:
        age_ans = 'Adult'
      else:
        age_ans = 'Young'
      '''
      final_messg = str(f'It seems like you are {str(mf_ans)} with a/an {str(accent_ans)} accent.')
      
      '''
      American - 0
      other - 5
      British - 2
      Europian - 3
      Indian - 4
      Austrialian - 1
      '''
      '''
      Adult - 1
      Youth - 0
      '''
      print(final_messg)  
      resp = MessagingResponse()
      resp.message(final_messg)
    
      return str(resp)
    
    else:
      
      resp = MessagingResponse()
      final_messg = 'Try Again'
      resp.message(final_messg)
      
      return str(resp)

if __name__ == "__main__":
    app.run(debug=True)