import features
import dtw
import scipy.io.wavfile as wav


terms = ['assist', 'block', 'foul', 'make', 'miss', 'steal']
name = 'sam'
lowfreq=50
highfreq=8000
num_coefs=13
remove_coef1 = True
train_inds = [1,3,5,7,13]
test_inds = [2,8,12,14,15]

def gen_feats(term, ind):
   file_name = 'data/{}_{}{:02d}.wav'.format(name, term, ind)
   rate, sig = wav.read(file_name)
   feats = features.mfcc(sig, rate, numcep=num_coefs, lowfreq=lowfreq, highfreq=highfreq)
   if remove_coef1:
      feats = [lst[1:] for lst in feats]
   return feats

# collect training data
train_feats = {}
for term in terms:
   train_feats[term] = []
   for i in train_inds:
      feats = gen_feats(term, i)
      train_feats[term].append(feats)

print('done reading training data')

# test
for term in terms:
   for i in test_inds:
      feats = gen_feats(term, i)
      scores = {}
      for t in terms:
         scores[t] = []
         train_set = train_feats[t]
         for e in train_set:
            score = dtw.dtw(feats, e, dtw.list_distance)
            scores[t].append(score)

      # print(scores)
      min_scores = {k:min(v) for k,v in scores.items()}
      avg_scores = {k:sum(v)/len(v) for k,v in scores.items()}
      min_term, min_term_score = min(min_scores.items(), key= lambda tup: tup[1])
      min_avg_term, min_avg_term_score = min(avg_scores.items(), key= lambda tup: tup[1])

      print("{} {}: Closest: {} ({:.2f})  Min Avg: {} ({:.2f})".format(term, i, min_term, min_term_score, min_avg_term, min_avg_term_score))
      
