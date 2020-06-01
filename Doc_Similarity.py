from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import os
from azure_api import read_results
import csv

def get_textfiles(input_path, output_folder):
    file_path = input_path  # source path for the pdf
    folder_name = file_path.split("/")[-1]
    out_path = output_folder + folder_name + "/"  # destination path for output html files
    if not os.path.exists(out_path):  # Check if the directory exists
        os.makedirs(out_path)  # Create directory with the output path
    print(folder_name)
    if not len(os.listdir(os.path.join(output_folder,folder_name))):
        text_list = []
        print(os.listdir(input_path))
        for pdf_name in sorted(os.listdir(input_path), key=len):
            print(pdf_name)
            pdf_name = pdf_name.split(".pdf")[0]
            words, sentences = read_results(pdf_name)
            number = len(sentences)
            print(number)
            for i in range(number):
                text = " "
                for idx in range(len(sentences[i][1])):
                    text = text + " " + sentences[i][1][idx][0]
                with open(out_path + "{}.txt".format(pdf_name),
                          "a+") as file:  # If already a sentences.txt file exists, replace it by the new predictions
                    file.write(
                        text)  # Write the output sentences into a txt file in the directories that they belong to
                    file.write("\n")
            text_list.append([text])

def build_similarity(input_path, folder_name, document_name):
    file_docs = []
    files = []
    for i,file_name in enumerate(sorted(os.listdir(os.path.join(input_path, folder_name)), key=len)):
        if file_name.split(".txt")[0] == document_name: continue
        files.append(file_name.split(".txt")[0])
        with open(os.path.join(input_path,folder_name +"/" + file_name)) as f:
            text = f.read()
            filtered_text = remove_stopwords(text)
            tokens = word_tokenize(filtered_text)
        file_docs.append(tokens)
    # print(file_docs)
    # print(len(file_docs))
    gen_docs = file_docs
    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # print(corpus)
    tf_idf = gensim.models.TfidfModel(corpus)
    # for doc in tf_idf[corpus]:
    #     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
    sims = gensim.similarities.Similarity('./workdir/', tf_idf[corpus], num_features=len(dictionary))
    return sims, dictionary, tf_idf, files

def compare_docs(input_path, folder_name, document_name, output_path):
    sims, dictionary, tf_idf, files = build_similarity(input_path, folder_name, document_name)
    with open(os.path.join(input_path, folder_name + "/" + document_name + ".txt")) as f:
        text = f.read()
        filtered_text = remove_stopwords(text)
    query_doc = [w.lower() for w in word_tokenize(filtered_text)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # print('Comparing Result:', sims[query_doc_tf_idf]*100)
    values = sims[query_doc_tf_idf]
    result = {files[i]: round(values[i]*100,2) for i in range(len(values))}
    if not os.path.exists(os.path.join(output_path, folder_name)):
        os.makedirs(os.path.join(output_path, folder_name))
    output_path = os.path.join(output_path, folder_name)
    with open(output_path+"/"+document_name+".csv", 'w+',newline="") as f:
        field_names = ['Document Name', '% Similarity']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for key in result.keys():
            writer.writerow({'Document Name': key, '% Similarity': result[key]})


if __name__ == "__main__":
    compare_docs("../Data/Texts", "Covid_medicine", "Covid_Article_23", "../Output/Document_Similarity")
    # compare_docs()
    # get_textfiles("../Data/Pdfs/Covid_medicine", "../Data/Texts/")