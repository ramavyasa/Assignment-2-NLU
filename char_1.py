def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

raw_text = load_doc('gg_gut.txt')

tokens = raw_text.split()
raw_text = ' '.join(tokens)

length = 20
sequences = list()
for i in range(length, len(raw_text)):
	seq = raw_text[i-length:i+1]
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)