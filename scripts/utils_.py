# import torch
# from transformers import AutoModel, AutoTokenizer 



import socket
import random 

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def is_internet_available(host="8.8.8.8", port=53, timeout=1):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True 
    except socket.error as ex:
        print(ex)
        return False 


def index_generator(filename, length):
    max_digits = 7
    indices = []
    for idx in range(1, length + 1):
        digits_in_i = len(str(idx))

        if digits_in_i == max_digits:
            index = filename + "_" + str(i)
        else:
            padding_len = max_digits - digits_in_i
            index = filename + "_" + "0" * padding_len + str(idx)
        indices.append(index)

    return np.expand_dims(indices, axis=1)


def get_folds(df):
    """
    Return dataframe with a column indicating fold ID
    """
    L = list(range(1, 5)) 
    
    q = df.shape[0] // len(L) 
    r = df.shape[0] % len(L)
   
    
    folds = L * q + L[:r]
    
    random.seed(2020)
    random.shuffle(folds)
    
    # df["folds"] = folds
    
    return folds 



def get_TF_IDF_mat(list_of_tweets, list_of_IDs):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(list_of_tweets) # TODO - need to make it generalizable in future
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    
    TF_IDF_df = pd.DataFrame(denselist, columns=feature_names)
    TF_IDF_df["ID"] = list_of_IDs # n
    return TF_IDF_df


if __name__=="__main__":
    df = pd.read_csv("../data/stack_files/samp_raw_df_april.csv")
    df_TF_IDF = get_TF_IDF_mat(df["Tweet Text"], df["ID"])



    df = pd.read_csv("../data/stack_files/balanced_pro_n_anti_mask_df_v4.csv")
    TF_IDF_df = get_TF_IDF_mat(df["Tweet Text"], df["ID"])

    TF_IDF_df.to_csv("../data/stack_files/balanced_pro_n_anti_mask_TF_IDF_df_v4.csv")


# def main():
#     print(1)

#     return 0 

# is_internet_available()
# main()


# bertweet = AutoModel.from_pretrained("vinai/bertweet-baseipyth")
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# # INPUT TWEET IS ALREADY NORMALIZED!
# line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

# input_ids = torch.tensor([tokenizer.encode(line)])

# with torch.no_grad():
#     features = bertweet(input_ids)  # Models outputs are now tuples
    
# ## With TensorFlow 2.0+:
# # from transformers import TFAutoModel
# # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")


# import torch
# from transformers import AutoModel, AutoTokenizer 

# bertweet = AutoModel.from_pretrained("vinai/bertweet-covid19-base-cased")
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased")

# # INPUT TWEET IS ALREADY NORMALIZED!
# line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

# input_ids = torch.tensor([tokenizer.encode(line)])

# with torch.no_grad():
#     features = bertweet(input_ids)  # Models outputs are now tuples
    
# ## With TensorFlow 2.0+:
# # from transformers import TFAutoModel
# # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")



# import torch
# from transformers import AutoModel, AutoTokenizer 

# bertweet = AutoModel.from_pretrained("vinai/bertweet-covid19-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-uncased")

# # INPUT TWEET IS ALREADY NORMALIZED!
# line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

# input_ids = torch.tensor([tokenizer.encode(line)])

# with torch.no_grad():
#     features = bertweet(input_ids)  # Models outputs are now tuples
    
# ## With TensorFlow 2.0+:
# # from transformers import TFAutoModel
# # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")