import os 
import glob 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 


def tweet_lang(df):

	return df[df["Language"] == "en"].copy()


# def get_tweet_lang(data_dir, wrt_dir):
# 	files = glob.glob(os.path.join(data_dir, "*.csv"))

# 	total_num_eng_tweets = 0 
# 	for file in files:
# 		print("Filename:", file)
# 		df = pd.read_csv(file)
# 		print(f'Number of tweets {df.shape[0]}')

# 		df_eng = tweet_lang(df)

# 		total_num_eng_tweets += df_eng.shape[0]
# 		print(df_eng.shape)

# 		# write the file to the dir 
# 		eng_filename = file.split("/")[-1]
# 		df_eng.to_csv(os.path.join(wrt_dir, eng_filename), index=False, header=True)

# 	print("Total tweets:", total_num_eng_tweets)

# 	return 0 


def get_tweet_lang(data_dir, wrt_dir):
	files = glob.glob(os.path.join(data_dir, "*.csv"))

	total_num_eng_tweets = 0 
	filename_n_nTweets = {}
	for file in files:
		print("Filename:", file)
		df = pd.read_csv(file)
		print(f'Number of tweets {df.shape[0]}')

		df_eng = tweet_lang(df)

		total_num_eng_tweets += df_eng.shape[0]
		print(df_eng.shape)

		filename_n_nTweets[file.split("/")[-1]] = df_eng.shape[0] 

		# # write the file to the dir 
		# eng_filename = file.split("/")[-1]
		# df_eng.to_csv(os.path.join(wrt_dir, eng_filename), index=False, header=True)


	print("Total tweets:", total_num_eng_tweets)
	print("Filename and no of tweets:", filename_n_nTweets)

	return filename_n_nTweets


#TODO - get same number of tweets from promask 
def x():

	return 0 


if __name__ == "__main__":
	data_dir = "../data/NoMask_Tweets_Version_2"
	wrt_dir = "../data/nomask_tweets_v2_eng"

	_ = get_tweet_lang(data_dir, wrt_dir)


	# process promask tweets 
	data_dir = "../data/WearMask_Tweets"
	wrt_dir = "../data/stack_files"

	_ = get_tweet_lang(data_dir, wrt_dir)

	_ = get_tweet_lang(data_dir, wrt_dir)