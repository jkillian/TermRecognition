from record import wavrecord
from classifier import Classifier

if __name__ == '__main__':
    clf = Classifier()
    clf.train(plot=True)
    # clf.predict_word('')
    # clf.plot()


    # print("please speak a word into the microphone")
    # wavrecord.record_word_to_file('data/nate_block03.wav')
    # print("done - result written to demo.wav")

