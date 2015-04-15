from record import wavrecord

# wavrecord.record_for(5, 'data/output.wav')



if __name__ == '__main__':
    print("please speak a word into the microphone")
    wavrecord.record_to_file('demo.wav')
    print("done - result written to demo.wav")