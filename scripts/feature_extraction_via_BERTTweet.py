import os 
import glob 

import numpy as np 
import pandas as pd 

import torch
from transformers import AutoTokenizer
from transformers import AutoModel, AutoTokenizer


def get_BERTTweet_features(tweet):
	bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
	# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

	# Load the AutoTokenizer with a normalization mode if the input Tweet is raw
	tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

	# from transformers import BertweetTokenizer
	# tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

	# line = "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"

	input_ids = torch.tensor([tokenizer.encode(tweet)])


	with torch.no_grad():
	    feature_3D, feautre_2D = bertweet(input_ids)  # Models outputs are now tuples

	return feautre_2D.numpy().ravel().tolist()


def get_features(df, tweet_txt_col_idx):
	n_tweets = df.shape[0]
	L = []
	for i in range(n_tweets):
		print("=" * 80 )
		print("Tweet no.:", i + 1)
		print("Tweet:", df.iloc[i, tweet_txt_col_idx])
		feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
		L.append(feature)

	return pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))])


def wrt_feature_files(data_dir, wrt_dir):
	files = glob.glob(os.path.join(data_dir, "*.csv"))
	print(files)

	tweet_txt_col_idx = 0 # column no 0 is the tweet text column 

	for file in files:
		df = pd.read_csv(file)
		print(f'Number of tweets {df.shape[0]}')

		latent_val_df = get_features(df.iloc[:5, :], tweet_txt_col_idx)

		# write file to disk  
		latent_filename = file.split("/")[-1]
		latent_val_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)

	return 0 


if __name__ == "__main__":
	data_dir = "../data/NoMask_Tweets"
	wrt_dir = "../data/BERTTweet_Features"

	_ = wrt_feature_files(data_dir, wrt_dir)
