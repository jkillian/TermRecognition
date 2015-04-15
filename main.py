from record import wavrecord

# wavrecord.record_for(5, 'data/output.wav')



if __name__ == '__main__':
    print("please speak a word into the microphone")
    wavrecord.record_word_to_file('data/output.wav')
    print("done - result written to data/output.wav")