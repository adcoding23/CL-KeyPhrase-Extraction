# Keyphrase Extraction using Unsupervised Learning - EmbedRank with a variation
1. Download the StandfordCoreNLPTagger and start server: 
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
2. Download and install set2vec: https://github.com/epfml/sent2vec
3. Download one of the pre-trained models. I used wiki_unigrams.bin
4. Ensure fasttext.o is in the same directory as EmbedRank.py
