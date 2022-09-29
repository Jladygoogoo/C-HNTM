import gensim.downloader as api

print(api.load('glove-twitter-200', return_path=True))