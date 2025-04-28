import nltk     # Import the Natural Language Toolkit
from nltk.corpus import PlaintextCorpusReader   #Import the PlainTextCorpusReader Module

Corpus     = PlaintextCorpusReader('./CORPUS/', '.*')
rawText    = Corpus.raw()
tokens     = nltk.word_tokenize(rawText)
TextCorpus = nltk.Text(tokens)     

print ("Compiling First 100 Concordance Entries ...")
print("\n",TextCorpus.concordance('GUN',200,100))
print("\nTEST DONE")
