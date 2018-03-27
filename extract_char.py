from nltk.corpus import gutenberg
import re

train_data = list()
test_data = list()
data = list()
all_words = set()

def extract():
	f = gutenberg.sents("carroll-alice.txt")
	global data
	for i in f:
		sen = []
		for j in i:
			k = re.sub(r'[^\w\s]','',j.strip().lower())
			sen.append(k) if k is not '' else None
		data.append(' '.join(sen))
	
	data = '\n'.join(data)
	f = open("gg_gut.txt",'w')
	f.write(data)
	f.close()
	

			
extract()