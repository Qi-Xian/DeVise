if __name__ == '__main__':
    import sys, os
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 5:
        print "Usage: python countFreq.py -infile -synset_"
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
