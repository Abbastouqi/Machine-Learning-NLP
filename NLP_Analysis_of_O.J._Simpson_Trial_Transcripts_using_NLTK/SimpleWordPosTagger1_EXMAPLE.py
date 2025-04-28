'''
Simple NLTK Tagger
'''
# import the part of speech Tagger
# import the word tokenizer
from nltk import pos_tag, word_tokenize

# import and create a PrettyTable to hold the results
from prettytable import PrettyTable

# PARTS of SPEECH Lookup
POSTAGS = {
        'CC':   'conjunction',
        'CD':   'CardinalNumber',
        'DT':   'Determiner',
        'EX':   'ExistentialThere',
        'FW':   'ForeignWord',
        'IN':   'Preposition',
        'JJ':   'Adjective',
        'JJR':  'AdjectiveComparative',
        'JJS':  'AdjectiveSuperlative',
        'LS':   'ListItem',
        'MD':   'Modal',
        'NN':   'Noun',
        'NNS':  'NounPlural',
        'NNP':  'ProperNounSingular',
        'NNPS': 'ProperNounPlural',
        'PDT':  'Predeterminer',
        'POS':  'PossessiveEnding',
        'PRP':  'PersonalPronoun',
        'PRP$': 'PossessivePronoun',
        'RB':   'Adverb',
        'RBR':  'AdverbComparative',
        'RBS':  'AdverbSuperlative',
        'RP':   'Particle',
        'SYM':  'Symbol',
        'TO':   'to',
        'UH':   'Interjection',
        'VB':   'Verb',
        'VBD':  'VerbPastTense',
        'VBG':  'VerbPresentParticiple',
        'VBN':  'VerbPastParticiple',
        'VBP':  'VerbNon3rdPersonSingularPresent',
        'VBZ':  'Verb3rdPersonSingularPresent',
        'WDT':  'WhDeterminer',
        'WP':   'WhPronoun',
        'WP$':  'PossessiveWhPronoun',
        'WRB':  'WhAdverb'
        }

tbl = PrettyTable(["Word", "POS", "Details"])

# Tokenize a sentence 
sentence = "I will get voting rights to be a reality for everyone"
engText = word_tokenize(sentence)

# Tag each word with it's part of speech using the NLTK pos_tagger
tags = pos_tag(engText)

# Store each word, tag pair in the PrettyTable
for eachTag in tags:
    word = eachTag[0]
    pos  = eachTag[1]
    details = POSTAGS[pos]  
    tbl.add_row([word, pos, details])
    
print("\n","Original Sentence:\n", sentence)
tbl.align = 'l'
print(tbl.get_string())
    

