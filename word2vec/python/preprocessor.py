#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

#this is for sinularize and synonyms
from pattern.en import singularize,wordnet,suggest
from collections import defaultdict
import os.path, pickle, logging, pdb, re
from gensim import utils, interfaces

logger = logging.getLogger(__name__)


class Preprocessor(interfaces.TransformationABC):
	def __init__(self,sentences,F,S):	
		self.learn_synsets(S);
		max_ngram = self.ngram_counter(self.synsets)
                pdb.set_trace()
		if not os.path.isfile(F):
			logging.info( "Write vocab in %s" %F)
			self.learn_vocab(sentences)
			f = open(F,'w')
			pickle.dump(self.vocab, f)
			f.close()
			logging.info( "Write syn_vocab in %s" %("syn_"+F))
			f = open("syn_"+F,'w')
			pickle.dump(self.synsets, f)
			f.close()
		else:
			print "%s is already exist" %F
			f = open(F)
			#load vocab
			self.vocab = pickle.load(f)
			f.close()
	def iswnid(self,word):
		length = len(word)
                if length != 9 : return False;
                check = 0;
                if word[0].isalpha(): check+=1
                for i in range(1,length):
                        if word[i].isdigit(): check+=1
                if( check == 9 ):
                        return True
                else:
                        return False

	def learn_synsets(self,file_name):
		logging.info("collection synsets which we wnat to maintain in word_space")
		self.synsets = defaultdict(int)
		self.words = []
		f = open(file_name,'r')
		line_no = 0;
		while True:
			word = f.readline();
			if not word: break
			word = word.rstrip();
			if self.iswnid(word): continue;
			#sing = singularize(word)
			#if not wordnet.synsets(sing): sing = word 
			if word in self.synsets: pass
			else:
			  self.synsets[word]= 0
			  self.words.append(word)
			if line_no % 10000 == 0:
                                logger.info("PROGRESS: at line #%i, learn %i word types" %
                                        (line_no, len(self.synsets)))
			line_no+=1 
                  
		#pdb.set_trace()
	        logger.info("PROGRESS: at line #%i, learn %i word types" %(line_no, len(self.synsets)))
		f.close()
	def learn_vocab(self,sentences ):
        	
        	self.total_words = 0
        	logger.info("collecting all words and their counts")
        	self.vocab = defaultdict(basestring)
		#pdb.set_trace()
        	for sentence_no, sentence in enumerate(sentences):
            		if sentence_no % 10000 == 0:
                		logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            		(sentence_no, self.total_words, len(self.vocab)))	
            		sentence = [utils.any2utf8(w) for w in sentence]
			#pdb.set_trace()
            		for word_a, word_b in zip(sentence, sentence[1:]):
				word_a = re.sub("[^a-zA-Z]+", "", word_a )
				word_b = re.sub("[^a-zA-Z]+", "", word_b )
				if not word_a: continue
				if self.predict_bigram(word_a,word_b):
					phrase = word_a + "_" + word_b
					self.add_vocab(phrase)
				self.add_vocab(word_a)
			
			#pdb.set_trace()
            		if sentence:    # add last word skipped by previous loop
                		word = sentence[-1]
				word = re.sub("[^a-zA-Z]+", "", word )
				if not word: continue
				self.add_vocab(word)

        	logger.info("collected %i word types from a corpus of %i words ( unigram ) and %i sentences" %
                    		(len(self.vocab), self.total_words, sentence_no + 1))

	def add_vocab(self,word):
		 if word in self.synsets: 
			self.synsets[word]+=1 #count frequency
		 	if word in self.vocab:
				self.total_words+=1
			else:
				self.vocab[word] = word
				self.total_words+=1
			pass
		 if word in self.vocab:
			self.total_words+= 1
			pass
                 elif not word in self.synsets:
                 	singular = singularize(word)
                        if not wordnet.synsets(singular): singular = word
                        self.vocab[word] = singular
                        self.total_words+= 1
                 elif word in self.synsets:
                        self.vocab[word] = word
                        self.total_words+= 1
	

	def predict_bigram(self,word_a,word_b):
		phrase = word_a + "_" + word_b;
		if phrase in self.synsets:
			return True
		else: 
			return False

	def ngram_counter(self,wlist):
		max_ngram = defaultdict(int)
		for w in wlist:
		 if '_' in w:
		  count = 0
		  for s in w:
		    if s == '_': count+=1
 		  max_ngram[w] = count
		return max_ngram
	
	def prep_text(self,p_num, sentences,outfile):
		output = open(outfile,'w')
		#distribute textfiles
		for i,sen in enumerate(sentences):
			sentence = [utils.any2utf8(w) for w in sen ]
			for word_a, word_b in zip(sentence, sentence[1:]):
				word_a = re.sub("[^a-zA-Z]+", "", word_a )
				word_b = re.sub("[^a-zA-Z]+", "", word_b )
				if not word_a: continue
				phrase = word_a + "_" + word_b;
        		        if phrase in self.vocab:
					output.write(utils.to_utf8(self.vocab[phrase]+' '))
				else:
					output.write(utils.to_utf8(self.vocab[word_a]+' '))
		
			if i % 10000 == 0:
        			logger.info("PROGRESS: at sentence #%i " %(i))
        	logger.info("PROGRESS: at sentence #%i " %(i))
    		output.close()

class util:
	#readwnids
	def CountFreq(self,file_name):
		f = open(file_name,'r')
                line_no = 0;
		freq = defaultdict(int)
		counts = 0
                while True:
                        word = f.readline();
                        if not word: break
                        word = word.rstrip();
                        if self.iswnid(word):
				freq[word]+=counts
				counts =0
				continue
			counts+=self.synsets[word]	
                        if line_no % 10000 == 0:
                                logger.info("PROGRESS: at line #%i" %
                                        (line_no ))
                        line_no+=1
		f.close()
		f = open( "sorted.txt",'w')
                logger.info("PROGRESS: at line #%i" %(line_no))
		for w in sorted(freq, key=freq.get, reverse=True):
		   sen  = "%s: %d" %(w,freq[w])
		   f.write( sen+'\n')
		f.close()
	def load_synset(self,S):
		f = open(S,'r')
		self.synsets = pickle.load(f)
		f.close()

        def iswnid(self,word):
                length = len(word)
                if length != 9 : return False;
                check = 0;
                if word[0].isalpha(): check+=1
                for i in range(1,length):
                        if word[i].isdigit(): check+=1
                if( check == 9 ):
                        return True
                else:
                        return False
		

if __name__ == '__main__':
    import sys, os
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 5:
        print "Usage: python preprocessor.py -infile -synset_list -vocab_filename -outputfilename "
        sys.exit(1)
    infile = sys.argv[1]
    #synset filename
    S = sys.argv[2]
    #vocab filename
    F = sys.argv[3]
    outfile = sys.argv[4]

    from preprocessor import Preprocessor  # for pickle
    #from gensim.models import Preprocessor  # for pickle
    from gensim.models.word2vec import Text8Corpus
    sentences = Text8Corpus(infile)

    prep = Preprocessor(sentences,F,S)
    prep.prep_text(8,sentences,outfile)
