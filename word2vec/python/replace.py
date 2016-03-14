#!/usr/bin/env python
import pdb,sys,os.path,re,wikipedia,logging,urllib2,requests
from collections import defaultdict
from wikipedia import search
from pattern.en import singularize,pluralize,wordnet
reload(sys)
sys.setdefaultencoding('utf8')

root_path = '/v2-1/word2vec/'
#root_path = ''
debug_mode = 1; 
logger = logging.getLogger(__name__)

class replacer:
	def __init__(self,infile,debug):
		filename = root_path + infile;
		logger.info("#File = %s #Debug = %i" %(filename,debug))
		self.fin = open(filename,'r')
		self.debug = debug
		self.trial_mode = 5;
		if self.debug:
			 self.dfile = open("debug3.txt",'w')
	def purify(self,slabel):
		if '(' in slabel:
			nlist = slabel.split('(')
			slabel = nlist[0].rstrip()
		if ',' in slabel:
			nlist = slabel.split(',')
			slabel = nlist[0].rstrip()
		if ' ' in slabel:
			slabel = re.sub(" ", "_", slabel )
		return slabel;

	def replace(self,outfile):
		filename = root_path + outfile
		logger.info("#output = %s" %filename)
		f = open(filename,'w')
		x = 0
		vocab = defaultdict(basestring)
		while True:
			label = self.fin.readline()
			label = label.rstrip()
			if not label: break
			f.write(label+'\n')
			#to avoid duplicated words at each catergory
			vocab[label] = True
			if self.iswnid(label): 
				vocab.clear()
				x+=1
				if( x%100 == 0):
					logger.info("Progress .. #%i" %x)
				continue
			#check wikipedia article redirect
			word = self.suggest(re.sub("[^a-zA-Z-]+", " ",label))
                        origin = word.encode().lower()
			slabel = singularize(word.encode().lower())
			plabel = pluralize(word.encode().lower())
			if not wordnet.synsets(slabel): nlabel = word.encode().lower()
			slabel = self.purify(slabel);
			plabel = self.purify(plabel);

			if not origin in vocab:
				f.write(origin+'\n')
				vocab[origin] = True
			if not slabel in vocab:
				f.write(slabel+'\n')
				vocab[slabel] = True
			if not plabel in vocab:
				f.write(plabel+'\n')
				vocab[plabel] = True
		f.close()
		if self.debug:
			self.dfile.close()
		print "Done.."

	def confirm(self,outfile):
                x = 0
		check = 0
		filename = root_path + outfile
		logger.info("#output = %s" %filename)
		f = open(filename,'w')
		while True:
                        label = self.fin.readline()
                        label = label.rstrip()
                        if not label: break
			if self.iswnid(label): 
				x+=1
				if( x%10 == 0):
					logger.info("Progress .. #%i" %x)
				line = "%s %d\n" %(label,check)
				f.write(line)
				check = 0
				continue
			#check wikipedia article redirect
			success = self.isPage(re.sub("[^a-zA-Z-]+", " ",label))
                	if not success and self.debug:
				self.dfile.write(label+'\n')
			if success and not check:
				check = 1
		f.close()
		if self.debug:
			self.dfile.close()
		print "Done.."

    
	def suggest(self,word):
		success,nword = self.isPage(word)
		return nword

	def isPage(self,word):
		try_num = self.trial_mode
		issuccess = False
		try:
		 	s = wikipedia.WikipediaPage(word)
                        word = s.title;
			issuccess =True
		except wikipedia.exceptions.DisambiguationError as e:
			 #logger.info("%s is in DisambiguationError " %word)
			if len(e.options) > 0:
			 	word = str(e.options[0]);
			else: pass
		except wikipedia.exceptions.HTTPTimeoutError as e:
  			logger.info("%s is in HTTPTimeoutError " %word)
			while try_num:
				try_num-=1
		 		s = wikipedia.WikipediaPage(word)
				issuccess=True			
		except wikipedia.exceptions.PageError as e: pass
		except: pass
		return issuccess,word
	
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

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.CRITICAL)
    logging.info("running %s" % " ".join(sys.argv))
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print "Usage: python replace.py infile  outfile mode(default = 0)"
        sys.exit(1)
    infile = sys.argv[1]
    outfile =  sys.argv[2] 
    if len(sys.argv) == 4:
	mode = int(sys.argv[3])
    else:
	mode = 0

    wikipedia.set_rate_limiting(1);    
    from replace import replacer
    rep = replacer(infile,debug_mode)
    if not mode :
     logging.info( "Usage: replacer")
     rep.replace(outfile)
    else:
     logging.info( "Usage: confirm" )
     rep.confirm(outfile)
   
