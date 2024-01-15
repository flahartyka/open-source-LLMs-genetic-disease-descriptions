
import pandas as pd
import csv
import mediawiki
from mediawiki import MediaWiki
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path  
import glob
import numpy as np
import heapq
from tensorflow.python.ops.numpy_ops import np_config 
from sentence_transformers import SentenceTransformer
np_config.enable_numpy_behavior()


def get_corr(first, second): 
    corr = np.inner(first, second)
    return corr


#-----------------------------------------------------------------#
#LISTS

#TFIDF WIKIPEDIA
tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')

message_df = pd.read_csv('wiki_diseases_download.csv')
definition_list = message_df["Wiki Article"].tolist()
name_list = message_df["Disease Name"].tolist()

query_file = open("medical_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    message_df = pd.read_csv('wiki_diseases_download.csv')
    definition_list = message_df["Wiki Article"].tolist()
    name_list = message_df["Disease Name"].tolist()
    query_name= 'Query'
    definition_list.append(query)
    name_list.append(query_name)
    content = definition_list
    tfidf_vector = tfidf_vectorizer.fit_transform(content)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=name_list, columns=tfidf_vectorizer.get_feature_names_out())
    full_array = tfidf_vector.toarray()
    qq = full_array[len(full_array) - 1]
    scores = []
    for entry in full_array:
        score = get_corr(entry, qq)
        scores.append(score)
    scores.pop()
    N = 50
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = name_list[num]
        if first == 1:
            string = update
        else:
            string = string + ', ' + update
        first = 0
    strings.append(string)


df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('wiki_med_tfidf_def.csv')

## CREATE Vicuna File
df = pd.read_csv('wiki_med_tfidf_def.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"


i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('med_wiki_tfidf_vicuna_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('wiki_med_tfidf_def.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('med_wiki_tfidf_llama_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()








# TFIDF MEDLINE

def get_corr(first, second): 
    corr = np.inner(first, second)
    return corr


message_df = pd.read_csv('example_medline.csv')
definition_list = message_df["Disease Definition"].tolist()
name_list = message_df["Disease Name"].tolist()

query_file = open("layman_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    message_df = pd.read_csv('example_medline.csv')
    definition_list = message_df["Disease Definition"].tolist()
    name_list = message_df["Disease Name"].tolist()
    query_name= 'Query'
    definition_list.append(query)
    name_list.append(query_name)
    content = definition_list
    tfidf_vector = tfidf_vectorizer.fit_transform(content)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=name_list, columns=tfidf_vectorizer.get_feature_names_out())
    full_array = tfidf_vector.toarray()
    qq = full_array[len(full_array) - 1]
    scores = []
    for entry in full_array:
        score = get_corr(entry, qq)
        scores.append(score)
    scores.pop()
    N = 50
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = name_list[num]
        if first == 1:
            string = update
        else:
            string = string + ', ' + update
        first = 0
    strings.append(string)

df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('medline_lay_tfidf_names.csv')

## CREATE Vicuna File
df = pd.read_csv('medline_lay_tfidf_names.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('lay_medline_tfidf_vicuna.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('medline_lay_tfidf_names.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('lay_medline_tfidf_llama.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()








#ROBERTA MEDLINE


#load embeddings model
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

#embedding definition
def embed(input):
  return model.encode(input)

message_df = pd.read_csv('example_medline.csv')
definition_list = message_df["Disease Definition"].tolist()
messages = definition_list
name_list = message_df["Disease Name"].tolist()

message_embeddings = embed(messages)
message_embedding_list = message_embeddings.tolist()

#format embeddings/messages in a dataframe
df = pd.DataFrame(list(zip(messages, message_embedding_list, name_list)), columns = ['Messages', 'Embeddings', 'Disease Name'])

#save this dataframe so we only have to do this once
df.to_csv('medline_embeddings.csv')

embedding_list = df.Embeddings.values.tolist()
message_list = df.Messages.values.tolist()
name_list = df['Disease Name'].tolist()

query_file = open("layman_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    q_embed = embed(query)
    scores = []
    for entry in embedding_list:
        score = get_corr(entry, q_embed)
        scores.append(score)
    N = 50
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = name_list[num]
        if first == 1:
            string = update
        else:
            string = string + ', ' + update
        first = 0
    strings.append(string)


df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('medline_lay_roberta_names.csv')

## CREATE Vicuna File
df = pd.read_csv('medline_lay_roberta_names.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('lay_medline_roberta_vicuna.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('medline_lay_roberta_names.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('lay_medline_roberta_llama.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

















#ROBERTA WIKIPEDIA

message_df = pd.read_csv('wiki_diseases_download.csv')
definition_list = message_df["Wiki Article"].tolist()
name_list = message_df["Disease Name"].tolist()

messages= definition_list

message_embeddings = embed(messages)
message_embedding_list = message_embeddings.tolist()

#format embeddings/messages in a dataframe
df = pd.DataFrame(list(zip(messages, message_embedding_list, name_list)), columns = ['Messages', 'Embeddings', 'Disease Name'])


embedding_list = df.Embeddings.values.tolist()
message_list = df.Messages.values.tolist()
name_list = df['Disease Name'].tolist()

query_file = open("layman_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    q_embed = embed(query)
    scores = []
    for entry in embedding_list:
        score = get_corr(entry, q_embed)
        scores.append(score)
    N = 50
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = name_list[num]
        if first == 1:
            string = update
        else:
            string = string + ', ' + update
        first = 0
    strings.append(string)


df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('wiki_lay_roberta_names.csv')

## CREATE Vicuna File
df = pd.read_csv('wiki_lay_roberta_names.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('lay_wiki_roberta_vicuna.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('wiki_lay_roberta_names.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('lay_wiki_roberta_llama.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()














#DEFINITONS 
#--------------------------------------------------------------


#TFIDF WIKIPEDIA
tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')

message_df = pd.read_csv('wiki_diseases_download.csv')
definition_list = message_df["Wiki Article"].tolist()
name_list = message_df["Disease Name"].tolist()
summary_list = message_df["Summary"].tolist()

query_file = open("layman_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    message_df = pd.read_csv('wiki_diseases_download.csv')
    definition_list = message_df["Wiki Article"].tolist()
    name_list = message_df["Disease Name"].tolist()
    summary_list = message_df["Summary"].tolist()
    query_name= 'Query'
    definition_list.append(query)
    name_list.append(query_name)
    content = definition_list
    tfidf_vector = tfidf_vectorizer.fit_transform(content)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=name_list, columns=tfidf_vectorizer.get_feature_names_out())
    full_array = tfidf_vector.toarray()
    qq = full_array[len(full_array) - 1]
    scores = []
    for entry in full_array:
        score = get_corr(entry, qq)
        scores.append(score)
    scores.pop()
    N = 5
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = summary_list[num]
        if first == 1:
            string = update
        else:
            string = string + '-n ' + update
        first = 0
    strings.append(string)


df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('wiki_lay_tfidf_def.csv')

## CREATE Vicuna File
df = pd.read_csv('wiki_lay_tfidf_def.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('lay_wiki_tfidf_vicuna_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('wiki_lay_tfidf_def.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('lay_wiki_tfidf_llama_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()










# TFIDF MEDLINE

def get_corr(first, second): 
    corr = np.inner(first, second)
    return corr


message_df = pd.read_csv('example_medline.csv')
definition_list = message_df["Disease Definition"].tolist()
name_list = message_df["Disease Name"].tolist()

query_file = open("medical_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    message_df = pd.read_csv('example_medline.csv')
    definition_list = message_df["Disease Definition"].tolist()
    name_list = message_df["Disease Name"].tolist()
    query_name= 'Query'
    definition_list.append(query)
    name_list.append(query_name)
    content = definition_list
    tfidf_vector = tfidf_vectorizer.fit_transform(content)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=name_list, columns=tfidf_vectorizer.get_feature_names_out())
    full_array = tfidf_vector.toarray()
    qq = full_array[len(full_array) - 1]
    scores = []
    for entry in full_array:
        score = get_corr(entry, qq)
        scores.append(score)
    scores.pop()
    N = 5
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = definition_list[num]
        if first == 1:
            string = update
        else:
            string = string + '-n' + update
        first = 0
    strings.append(string)

df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('medline_med_tfidf_def.csv')

## CREATE Vicuna File
df = pd.read_csv('medline_med_tfidf_def.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following examples, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. "
last_part = "/nGiven what you know about all these genetic conditions, what is the most likely genetic diagnosis?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('med_medline_tfidf_vicuna_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('medline_med_tfidf_def.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. "
last_part = "/nGiven what you know about all these genetic conditions, what is the most likely genetic diagnosis?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('med_medline_tfidf_llama_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()










#ROBERTA MEDLINE


#load embeddings model
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

#embedding definition
def embed(input):
  return model.encode(input)

message_df = pd.read_csv('example_medline.csv')
definition_list = message_df["Disease Definition"].tolist()
messages = definition_list
name_list = message_df["Disease Name"].tolist()

message_embeddings = embed(messages)
message_embedding_list = message_embeddings.tolist()

#format embeddings/messages in a dataframe
df = pd.DataFrame(list(zip(messages, message_embedding_list, name_list)), columns = ['Messages', 'Embeddings', 'Disease Name'])
savedf = df 
#save this dataframe so we only have to do this once
df.to_csv('medline_embeddings.csv')

embedding_list = df.Embeddings.values.tolist()
message_list = df.Messages.values.tolist()
name_list = df['Disease Name'].tolist()

query_file = open("medical_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    q_embed = embed(query)
    scores = []
    for entry in embedding_list:
        score = get_corr(entry, q_embed)
        scores.append(score)
    N = 5
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = message_list[num]
        if first == 1:
            string = update
        else:
            string = string + '-n ' + update
        first = 0
    strings.append(string)


df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('medline_med_roberta_def.csv')

## CREATE Vicuna File
df = pd.read_csv('medline_med_roberta_def.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('med_medline_roberta_vicuna_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('medline_med_roberta_def.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('med_medline_roberta_llama_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

















#ROBERTA WIKIPEDIA

message_df = pd.read_csv('wiki_diseases_download.csv')
definition_list = message_df["Wiki Article"].tolist()
name_list = message_df["Disease Name"].tolist()
summary_list = message_df["Summary"].tolist()

messages= definition_list

message_embeddings = embed(messages)
message_embedding_list = message_embeddings.tolist()

#format embeddings/messages in a dataframe
df = pd.DataFrame(list(zip(messages, message_embedding_list, name_list)), columns = ['Messages', 'Embeddings', 'Disease Name'])


embedding_list = df.Embeddings.values.tolist()
message_list = df.Messages.values.tolist()
name_list = df['Disease Name'].tolist()

query_file = open("layman_query.txt", "r")
data = query_file.read()
query_list = data.split('\n')

string = ""
strings = []
scores = []

for query in query_list:
    q_embed = embed(query)
    scores = []
    for entry in embedding_list:
        score = get_corr(entry, q_embed)
        scores.append(score)
    N = 5
    res = [scores.index(i) for i in heapq.nlargest(N, scores)]
    string = ""
    first = 1
    for num in res:
        update = summary_list[num]
        if first == 1:
            string = update
        else:
            string = string + '-n ' + update
        first = 0
    strings.append(string)


df = pd.DataFrame(list(zip(query_list, strings)))

df.to_csv('wiki_lay_roberta_def.csv')

## CREATE Vicuna File
df = pd.read_csv('wiki_lay_roberta_def.csv')
strings = df.iloc[:,2].tolist()
id_num = 1
first_part = "'category': 'stem', 'turns': ['Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples. /nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?', '']}"

#string = strings[0]
i = 0
questions = []
for string in strings:
    line = "{'question_id': " + str(id_num) + ", " + first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    id_num = id_num+1
    questions.append(line)

file = open('lay_wiki_roberta_vicuna_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

##CREATE Llama File
df = pd.read_csv('wiki_lay_roberta_def.csv')
strings = df.iloc[:,2].tolist()
first_part = "[{'role': 'user', 'content': 'Given the following conditions, predict the most likely genetic diagnosis. The most likely diagnosis may not be in this list of examples./nConditions: "
last_part = "/nGiven what you know about all these genetic conditions, what are the top 5 most likely genetic diagnoses?'}],"

i = 0
questions = []
for string in strings:
    line = first_part + string + "/n/n" + query_list[i] + last_part
    i = i+1
    questions.append(line)

file = open('lay_wiki_roberta_llama_def.jsonl','w')
for item in questions:
	file.write(item+"\n")

file.close()

