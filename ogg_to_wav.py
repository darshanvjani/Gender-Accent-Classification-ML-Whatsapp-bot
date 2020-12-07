
from pydub import AudioSegment

ogg_version = AudioSegment.from_ogg('F:/Gender_Accent_classification/jani.ogg')
ogg_version.export("F:/Gender_Accent_classification/wav_1.wav", format="wav")

'''import os

path = os.path.dirname(os.path.abspath(__file__))
print(path)'''