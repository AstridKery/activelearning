import util
import sys

st = sys.argv
fName = str(st[1])
if "lemm" in st:
    lemm=True
else:
    lemm=False

if "stemm" in st:
    stemm=True
else:
    stemm=False

if "stop" in st:
    stop=True
else:
    stop=False
#fName = "Batch_3166944_batch_results.csv"
#fName = "Batch_3169679_batch_results.csv"
#fName = "Batch_3178122_batch_results.csv"
util.fileringSentences(fName,lemm,stemm,stop)
