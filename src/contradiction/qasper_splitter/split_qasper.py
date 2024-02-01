from typing import List
import nltk

class Splitter:
    '''A class for splitting data in the qasper dataset
    '''
    sentances = []
    words = []

    def __init__(self, data) -> None:
        '''Initializes class by loading the json data'''
        self.data = data

    def split_paragraphs_to_sentances(self) -> (List[str], List[str]):
        '''splits all the paragraphs to sentances'''
        
        for paper in self.data:
            for full_text in self.data[paper]['full_text']:
                for paragraph in full_text['paragraphs']:
                    self.sentances += nltk.sent_tokenize(paragraph)
                    self.words += [word for word in nltk.word_tokenize(paragraph) if word.isalpha()]
        return self.sentances, self.words