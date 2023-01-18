"""
Mingi Kang 
January 16, 2022 
Professor Kumar 
Project : Lexicon Lab Coding Exercises
"""

import pandas as pd 
import matplotlib as mp 
import numpy as np
from numpy.linalg import norm 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from sklearn import datasets 
from switch import switch_simdrop 


class Similarity: 
    
    def __init__(self): 
        """Dictionary with ID and Animals Produced as Keys and Values from data-cochlear.txt"""
        self.data_cochlear = {} 
        with open('data-cochlear.txt', 'r') as contents: 
            data = contents.readlines() 
        for lines in data: 
            lines = lines.strip()
            lines = lines.replace('\t', ' ') 
            lines = lines.split() 
            if lines[0] in self.data_cochlear.keys(): 
                self.data_cochlear[lines[0]] += [lines[1]]
            else: 
                self.data_cochlear[lines[0]] = [lines[1]]
                
        """Word2vec dictionary with Word and Embedding as Keys and Values from word2vec.txt"""
        self.word2vec = {} 
        with open('word2vec.txt', 'r') as contents: 
            data = contents.readlines()
            data = data[1:]
        for lines in data: 
            lines = lines.strip() 
            lines = lines.split() 
            self.word2vec[lines[0]] = [eval(i) for i in list(lines[1:])] 
        
        """Speech2vec dictionary with Word and Embeddings as Keys and Values from speech2vec.txt"""
        self.speech2vec = {} 
        with open('speech2vec.txt', 'r') as contents: 
            data = contents.readlines()
            data = data[1:]
        for lines in data: 
            lines = lines.strip()
            lines = lines.split() 
            self.speech2vec[lines[0]] = [eval(i) for i in list(lines[1:])]
        
        """New Dictionaries of participant data without words that are not in word2vec.txt and speech2vec.txt"""
        self.w2v_words = {} 
        for id in self.data_cochlear: 
            words = [] 
            for word in self.data_cochlear[id]: 
                try: 
                    self.word2vec[word]
                    words.append(word) 
                except KeyError: 
                    pass 
            self.w2v_words[id] = words
            
        self.s2v_words = {} 
        for id in self.data_cochlear: 
            words = [] 
            for word in self.data_cochlear[id]: 
                try: 
                    self.speech2vec[word]
                    words.append(word)
                except KeyError: 
                    pass 
            self.s2v_words[id] = words 
            
        """Dictionary with ID and Embeddings as Keys and Values"""
        self.w2v_embeddings = {} 
        for id in self.w2v_words: 
            embeddings = [] 
            for word in self.w2v_words[id]: 
                embeddings.append(self.word2vec[word]) 
            self.w2v_embeddings[id] = embeddings 
            
        self.s2v_embeddings = {} 
        for id in self.s2v_words: 
            embeddings = [] 
            for word in self.s2v_words[id]: 
                embeddings.append(self.speech2vec[word]) 
            self.s2v_embeddings[id] = embeddings 
    
    def cosine_similarity(self, word1, word2, model): 
        if word1 == word2: 
            return 1
        else: 
            if model == 'word2vec': 
                A = np.array(self.word2vec[word1]) 
                B = np.array(self.word2vec[word2]) 
                return np.dot(A,B) / (norm(A) * norm(B))
            else: 
                A = np.array(self.speech2vec[word1]) 
                B = np.array(self.speech2vec[word2])
                return np.dot(A,B) / (norm(A) * norm(B)) 
        
    def visualize_items(self, ID): 
        tsne_2d = TSNE(perplexity= 20, n_components= 2, init= 'pca', n_iter= 3500, random_state= 32) 
        w2v_embeddings_2d = tsne_2d.fit_transform(np.array(self.w2v_embeddings[ID]))
        s2v_embeddings_2d = tsne_2d.fit_transform(np.array(self.s2v_embeddings[ID])) 
        
        words = [] 
        plt.figure(figsize = (16, 9)) 
        plt.title(f"{ID} T-SNE Plot")
        x = w2v_embeddings_2d[:,0] 
        y = w2v_embeddings_2d[:,1] 
        plt.scatter(x, y, c= 'r', alpha= 0.8, label = f"{ID} Word2Vec") 
        a = s2v_embeddings_2d[:,0] 
        b = s2v_embeddings_2d[:,1] 
        plt.scatter(a, b, c= 'k', alpha= 0.8, label = f"{ID} Speech2Vec") 
        for i, word in enumerate(words): 
            plt.annotate(word, alpha = 0.5, xy= (x[i], y[i]), xytext= (5,2), textcoords= 'offset points', ha= 'right', va= 'bottom', size = 10) 
        for i, word in enumerate(words): 
            plt.annotate(word, alpha = 0.5, ab= (x[i], y[i]), abtext= (5,2), textcoords= 'offset points', ha= 'right', va= 'bottom', size = 10) 
        plt.legend(loc= 4) 
        plt.grid(True) 
        plt.savefig(f'{ID}_visualize_items.png', format= 'png', dpi= 150, bbox_inches= 'tight') 
        plt.show() 
        
    def pairwise_similarity(self): 
        
        self.w2v_scores = {} 
        self.s2v_scores = {}    
        
        for ID in self.w2v_words: 
            num = [2] 
            idx = 1 
            while idx != len(self.w2v_words[ID]):
                num += [self.cosine_similarity(self.w2v_words[ID][idx -1], self.w2v_words[ID][idx], 'word2vec')] 
                idx += 1 
            self.w2v_scores[ID] = num 
            
        for ID in self.s2v_words: 
            num = [2] 
            idx =1 
            while idx != len(self.s2v_words[ID]): 
                num += [self.cosine_similarity(self.s2v_words[ID][idx-1], self.s2v_words[ID][idx], 'speech2vec')]
                idx += 1
            self.s2v_scores[ID] = num 
    
        data = {'ID': self.data_cochlear.keys(), 'Word2Vec Similarity': self.w2v_scores.values(), 'Speech2Vec Similarity': self.s2v_scores.values()}
        self.df = pd.DataFrame(data) 
        file_name = 'pairwise_similarity.csv'
        self.df.to_csv(file_name) 
        return self.df 
        
class Clusters(Similarity): 
    
        def compute_clusters(self): 
            
            a = Similarity()
            self.df = a.pairwise_similarity()
            self.w2v_clusters = {} 
            self.w2v_switches = {} 
            self.s2v_clusters = {} 
            self.s2v_switches = {} 
            
            for index, row in self.df.iterrows(): 
                simdrop = switch_simdrop(self.w2v_words[row['ID']], row['Word2Vec Similarity']) 
                clusters = [] 
                switches = [] 
                idx = 0 
                while idx < len(simdrop): 
                    if simdrop[idx] == 0: 
                        clusters.append(row['Word2Vec Similarity'][idx])
                        idx += 1 
                    elif simdrop[idx] == 1: 
                        switches.append(row['Word2Vec Similarity'][idx]) 
                        idx += 1
                    else: 
                        idx += 1
                self.w2v_clusters[row['ID']] = clusters 
                self.w2v_switches[row['ID']] = switches 
            
            for index, row in self.df.iterrows(): 
                simdrop = switch_simdrop(self.s2v_words[row['ID']], row['Speech2Vec Similarity'])
                clusters = [] 
                switches = [] 
                idx = 0 
                while idx < len(simdrop):
                    if simdrop[idx] == 0: 
                        clusters.append(row['Speech2Vec Similarity'][idx]) 
                        idx += 1
                    elif simdrop[idx] == 1: 
                        switches.append(row['Speech2Vec Similarity'][idx])
                        idx += 1
                    else: 
                        idx += 1
                self.s2v_clusters[row['ID']] = clusters 
                self.s2v_switches[row['ID']] = switches 
                
            data = {'ID': self.data_cochlear.keys(), 'W2V Words' : self.w2v_words.values(), 'W2V Clusters' : self.w2v_clusters.values(), "W2V Switches" : self.w2v_switches.values(), 'S2V Words' : self.s2v_words.values(), 'S2V Clusters' : self.s2v_clusters.values(), 'S2V Switches' : self.s2v_switches.values()}
            self.df = pd.DataFrame(data) 
            file_name = 'compute_clusters.csv'
            self.df.to_csv(file_name) 
            return self.df 
        
        def visualize_clusters(self, ID): 
            a = Clusters()
            df = a.compute_clusters()
            
            n = 2
            mean_switches = (float(df.loc[df['ID'] == ID]["W2V Switches"].apply(len)) / float(df.loc[df['ID'] == ID]["W2V Words"].apply(len)), float(df.loc[df['ID'] == ID]["S2V Switches"].apply(len)) / float(df.loc[df['ID'] == ID]["S2V Words"].apply(len)))
            mean_clusters = (float(df.loc[df['ID'] == ID]["W2V Clusters"].apply(len)) / float(df.loc[df['ID'] == ID]["W2V Words"].apply(len)), float(df.loc[df['ID'] == ID]["S2V Clusters"].apply(len)) / float(df.loc[df['ID'] == ID]["S2V Words"].apply(len)))
            
            fig, ax = plt.subplots()
            index = np.arange(n) 
            bar_width = 0.35 
            opacity = 0.50
            
            plot1 = plt.bar(index, mean_switches, bar_width, alpha= opacity, color= 'b', label= 'Switches') 
            plot2 = plt.bar(index + bar_width, mean_clusters, bar_width, alpha= opacity, color= 'r', label= 'Clusters') 
            plt.ylabel('mean')
            plt.title('Mean Number of Switches and Clusters')
            plt.xticks(index + bar_width, ('Word2Vec', 'Speech2Vec'))
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{ID}_visualize_clusters.png', format = 'png', dpi = 150, bbox_inches= 'tight')
            plt.show()
                       
                    
a = Similarity() 
a.cosine_similarity('the', 'or', 'word2vec') 
a.visualize_items('CAF-657') 
a.pairwise_similarity()

b = Clusters()
b.compute_clusters()
b.visualize_clusters('CAF-657')
