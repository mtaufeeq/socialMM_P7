# import torch
# from transformers import AutoModel, AutoTokenizer 



import socket

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



# def main():
# 	print(1)

# 	return 0 

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