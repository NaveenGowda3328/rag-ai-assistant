# import requests

# def create_embedding(text):
#     r = requests.post("http://localhost:11434/api/embeddings", json={
#         "model": "bge-m3",
#         "prompt": text
#     })

#     embedding = r.json()['embedding']          # This is all audios to json files converting code
#     return embedding


# a = create_embedding("Cat sat on the mat")
# print(a)

# import requests
# import os
# import json
# import pandas as pd

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })
#                                                                                                        # All Json files in Pandas Data frame..
#     embedding = r.json()["embeddings"] 
#     return embedding


# jsons = os.listdir("jsons")  # List all the jsons 
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
       
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk) 
# print(my_dicts)

# df = pd.DataFrame.from_records(my_dicts)
# print(df)
# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)

## OUT PUT...

# Creating Embeddings for 10_Video, Audio & Media in HTML.mp3.json
# Creating Embeddings for 11_Semantic Tags  in HTML.mp3.json
# Creating Embeddings for 12_Exercise 1 - Pure HTML Media Player.mp3.json
# Creating Embeddings for 13_Entities, Code tag and more on HTML.mp3.json
# Creating Embeddings for 14_Introduction to CSS.mp3.json
# Creating Embeddings for 15_Inline, Internal & External CSS.mp3.json
# Creating Embeddings for 16_Exercise 1 - Solution & Shoutouts.mp3.json
# Creating Embeddings for 17_CSS Selectors MasterClass.mp3.json
# Creating Embeddings for 18_CSS Box Model - Margin, Padding & Borders.mp3.json
# Creating Embeddings for 1_Installing VS Code & How Websites Work.mp3.json
# Creating Embeddings for 2_Your First HTML Website.mp3.json
# Creating Embeddings for 3_Basic Structure of an HTML Website.mp3.json
# Creating Embeddings for 4_Heading, Paragraphs and Links.mp3.json
# Creating Embeddings for 5_Image, Lists, and Tables in HTML.mp3.json
# Creating Embeddings for 6_SEO and Core Web Vitals in HTML.mp3.json
# Creating Embeddings for 7_Forms and input tags in HTML.mp3.json
# Creating Embeddings for 8_Inline & Block Elements in HTML.mp3.json
# Creating Embeddings for 9_Id & Classes in HTML.mp3.json
#      number                         title   start     end                                               text  chunk_id                                          embedding
# 0        10  Video, Audio & Media in HTML    0.00    5.60   In today's video, we will learn how audio, vi...         0  [-0.016527507, 0.027639396, -0.034291085, 0.03...
# 1        10  Video, Audio & Media in HTML    5.60   10.56   In our very first video, we saw that we can e...         1  [-0.008235201, 0.018632436, 0.0029347169, 0.05...
# 2        10  Video, Audio & Media in HTML   10.56   13.32   Today I will tell you how the video is auto-p...         2  [-0.034510233, -0.00035402848, -0.0540756, 0.0...
# 3        10  Video, Audio & Media in HTML   13.32   15.44                         how to keep a muted video,         3  [-0.021773636, -0.005614915, -0.02547062, -0.0...
# 4        10  Video, Audio & Media in HTML   15.44   19.88   and how you can enable and disable the video ...         4  [-0.03673615, -0.0034152383, -0.047707506, 0.0...
# ...     ...                           ...     ...     ...                                                ...       ...                                                ...
# 7247      9          Id & Classes in HTML  623.60  625.60                               access the playlist,      7247  [-0.0060391272, 0.009019411, -0.04789618, 0.01...
# 7248      9          Id & Classes in HTML  625.60  627.60                                     access GitHub,      7248  [-0.008582484, 0.013322549, -0.008749978, -0.0...
# 7249      9          Id & Classes in HTML  627.60  629.60                                    like the video,      7249  [0.0028567025, -0.0009740914, -0.075441524, 0....
# 7250      9          Id & Classes in HTML  630.40  632.40         Thank you so much for watching this video.      7250  [-0.039382737, 0.016052317, -0.0407894, 0.0232...
# 7251      9          Id & Classes in HTML  632.40  634.40                      And I will see you next time.      7251  [-0.006915486, 0.03867023, -0.038091585, 0.006...

# [7252 rows x 7 columns]


import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
    break
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts) 
incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
print(max_indx)
new_df = df.loc[max_indx] 
print(new_df[["title", "number", "text"]])