
import os
import sys
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from time import sleep
from prettytable import PrettyTable
from datetime import datetime

nltk.download('punkt')
nltk.download('stopwords')

stopSet = set(stopwords.words('english'))

# Output file setup
output_file = open("output_log.txt", "w", encoding="utf-8")
def log(text, print_also=True):
    if print_also:
        print(text)
    output_file.write(str(text) + '\n')

class classNLTKQuery:
    def textCorpusInit(self, thePath):
        if not os.path.isdir(thePath):
            return "Path is not a Directory"
        if not os.access(thePath, os.R_OK):
            return "Directory is not Readable"

        try:
            self.Corpus = PlaintextCorpusReader(thePath, '.*')
            log("📁 Processing Files:")
            log(self.Corpus.fileids())
            log("⏳ Please wait ...")
            self.rawText = self.Corpus.raw()
            self.tokens = nltk.word_tokenize(self.rawText)
            self.TextCorpus = nltk.Text(self.tokens)
        except:
            return "Corpus Creation Failed"

        self.ActiveTextCorpus = True
        return "Success"

    def printCorpusLength(self):
        log("\n📌 Corpus Text Length: {:,}".format(len(self.rawText)))

    def printTokensFound(self):
        log("\n📌 Tokens Found: {:,}".format(len(self.tokens)))

    def printVocabSize(self):
        log("\n📌 Calculating Vocabulary Size...")
        vocabularyUsed = set(self.TextCorpus)
        log("📌 Vocabulary Size: {:,}".format(len(vocabularyUsed)))

    def searchWordOccurrence(self):
        myWord = input("\n🔍 Enter Search Word: ").strip()
        log(f"\n🔍 Search Word: {myWord}")
        if myWord:
            wordCount = self.TextCorpus.count(myWord)
            log(f"📌 '{myWord}' occurred: {wordCount} times")
        else:
            log("⚠ No word entered.")

    def generateConcordance(self):
        myWord = input("\n🔍 Enter word for Concordance: ").strip()
        log(f"\n🔍 Concordance for: {myWord}")
        if myWord:
            log("📌 Concordance Entries:")
            from io import StringIO
            buffer = StringIO()
            sys.stdout = buffer
            self.TextCorpus.concordance(myWord, width=100, lines=100)
            sys.stdout = sys.__stdout__
            concord_output = buffer.getvalue()
            log(concord_output, print_also=False)
            print(concord_output)
        else:
            log("⚠ Invalid word.")

    def generateSimiliarities(self):
        myWord = input("\n🔍 Enter seed word for similarity: ").strip()
        log(f"\n🔍 Similar Words to: {myWord}")
        if myWord:
            from io import StringIO
            buffer = StringIO()
            sys.stdout = buffer
            self.TextCorpus.similar(myWord)
            sys.stdout = sys.__stdout__
            sim_output = buffer.getvalue()
            log(sim_output, print_also=False)
            print(sim_output)
        else:
            log("⚠ Invalid word.")

    def printWordIndex(self):
        myWord = input("\n🔍 Find index of what Word? : ").strip()
        log(f"\n🔍 Index of: {myWord}")
        if myWord:
            try:
                index = self.TextCorpus.index(myWord)
                log(f"📌 First occurrence at position: {index}")
            except ValueError:
                log("⚠ Word not found in corpus.")
        else:
            log("⚠ No word entered.")

    def printVocabulary(self):
        log("\n📌 Top Vocabulary Frequencies:")
        freqDist = nltk.FreqDist(self.TextCorpus)
        tbl = PrettyTable(["Word", "Occurrences"])
        for word, freq in freqDist.most_common(20):
            tbl.add_row([word, freq])
        tbl.align = 'l'
        log(tbl)

def printMenu():
    log("\n========= 🧠 NLTK Query Options =========")
    log("[1] Print Length of Corpus")
    log("[2] Print Number of Token Found")
    log("[3] Print Vocabulary Size")
    log("[4] Search for Word Occurrence")
    log("[5] Generate Concordance")
    log("[6] Generate Similarities")
    log("[7] Print Word Index")
    log("[8] Print Vocabulary")
    log("[0] Exit NLTK Experimentation")
    log("==========================================")

def getUserSelection():
    printMenu()
    while True:
        try:
            sel = input('\n🎯 Enter Selection (0-8): ')
            menuSelection = int(sel)
            log(f"🟢 Selected Option: {menuSelection}")
        except ValueError:
            log('❌ Invalid input. Enter a value between 0-8.')
            continue

        if not menuSelection in range(0, 9):
            log('❌ Invalid input. Enter a value between 0 - 8.')
            continue
        return menuSelection

if __name__ == '__main__':
    log(f"🕒 Script Run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("📚 Welcome to the NLTK Query Experimentation")
    log("🔁 Loading NLTK ...")

    print("\nInput path of corpus (e.g. CORPUS):")
    userSpecifiedPath = input("Path: ")
    log(f"📂 Corpus Path: {userSpecifiedPath}")

    oNLTK = classNLTKQuery()
    result = oNLTK.textCorpusInit(userSpecifiedPath)

    if result == "Success":
        menuSelection = -1
        while menuSelection != 0:
            if menuSelection != -1:
                input("\n👉 Press Enter to continue...")
                printMenu()

            menuSelection = getUserSelection()

            if menuSelection == 1:
                oNLTK.printCorpusLength()
            elif menuSelection == 2:
                oNLTK.printTokensFound()
            elif menuSelection == 3:
                oNLTK.printVocabSize()
            elif menuSelection == 4:
                oNLTK.searchWordOccurrence()
            elif menuSelection == 5:
                oNLTK.generateConcordance()
            elif menuSelection == 6:
                oNLTK.generateSimiliarities()
            elif menuSelection == 7:
                oNLTK.printWordIndex()
            elif menuSelection == 8:
                oNLTK.printVocabulary()
            elif menuSelection == 0:
                log("👋 Goodbye!")
                break
            sleep(2)
    else:
        log("❌ Failed to load corpus.")

    output_file.close()


    
    
