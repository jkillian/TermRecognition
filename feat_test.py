from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("data/output.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print(len(fbank_feat))
print(fbank_feat[300:303])