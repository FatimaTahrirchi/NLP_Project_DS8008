

import nltk
import string
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("averaged_perceptron_tagger")
nltk.download('tagsets')
nltk.download('treebank')
nltk.download('brown')
nltk.download('universal_tagset')

class Preprocessing():


    def __init__(self,Preprocess_Type='lower'):
        self.Preprocess_Type = Preprocess_Type
        self.stop_list = set(nltk.corpus.stopwords.words('english')+["#","@"])#english stop words
        self.temp_tag  = ['PRP','VBZ','CC','POS','IN','DT','TO','PRP$']#noise words
        self.stemmer   = nltk.stem.PorterStemmer()
        self.word_tokenize  = nltk.tokenize.word_tokenize
        self.pos_tag   = nltk.pos_tag
        self.punctuation = list(string.punctuation)
        self.neg_words = ["n't", "not", "no", "never"]#used in Add_Not preprocessing

    def transform(self, sentences):
        """
        sentences: list(str) 
        output: list(str)
        """
        if self.Preprocess_Type=='lower':
          return [s.lower() for s in sentences]
        elif self.Preprocess_Type=='denoiser':
          return [self.denoiser(s) for s in sentences]
        elif self.Preprocess_Type=='add_pos':
          return [self.add_pos(s) for s in sentences]
        elif self.Preprocess_Type=='add_hashtag':
          return [self.add_hashtag(s) for s in sentences]
        elif self.Preprocess_Type== 'add_NOT':
          return [self.add_Not(s) for s in sentences]


    def denoiser(self,text):
        new_text=""
        words=self.word_tokenize(text)
        words=nltk.pos_tag(words)
        words=[word.lower() for word,tag in words if tag not in self.temp_tag]#remove some extra words depends on their pos_tag
        words=[self.stemmer.stem(word) for word in words if word not in self.stop_list]#remove stop words and stemming
        for word in words:
            new_text=new_text+word+" "
        new_text=new_text[:-1]
        return new_text

    def add_pos(self,text):
        new_text=""
        words=self.word_tokenize(text)
        words=nltk.pos_tag(words)
        #concate pos_tag to the end of words,remove some extra words depends on their pos_tag,remove stop words and stemming
        words=[(self.stemmer.stem(word)+'@'+tag).lower() for word,tag in words if ((tag not in self.temp_tag) and (word.lower() not in self.stop_list))]
        for word in words:
            new_text=new_text+word+" "
        new_text=new_text[:-1]
        return new_text

    def add_hashtag(self,text):
        hashtaged = lambda word : '#'+word
        new_text=""
        words=self.word_tokenize(text)
        words=nltk.pos_tag(words)
        words2=[]
        hashtag=False
        for word,tag in words:
            if word=='#': 
                hashtag=True # if a previous token is '#' next token concated with '#' 
            if ((tag not in self.temp_tag) and (word not in self.stop_list)) or (word!='#' and hashtag==True):#remove some extra words depends on their pos_tag,remove stop words 
                if word[0].isupper():
                    hashtag=True # if a word is captalize will concate with '#'
                new_word=self.stemmer.stem(word).lower()# stemming
                if hashtag==False:
                    words2.append(new_word)
                if hashtag:
                    words2.append(hashtaged(new_word))
                    hashtag=False


        words=words2
        for word in words:
            new_text=new_text+word+" "
        new_text=new_text[:-1]
        return new_text

    def add_Not(self,text):
        new_text=""
        words= self.word_tokenize(text)
        words=self.pos_tag(words)
        words=[word.lower() for word,tag in words if tag not in self.temp_tag]#remove some extra words depends on their pos_tag 
        words=[self.stemmer.stem(word) for word in words ] 
        flag = 0  # start with the flag in the off position
        not_stem=[]
        for word in words:
            # if flag is on then append word with "NOT_"
            if flag == 1:
                # check if the word is a punctuation (this is where we need to stop if flag==1)
                if word in  self.punctuation:
                    # don't append anything to a punctuation
                    # if we reached here then it means the flag is to be turned off
                    not_stem.append(word)
                elif(word not in  self.neg_words):
                    not_stem.append("not_"+word)
                    
            # otherwise add the word without making any changes
            else:
                not_stem.append(word)
            
            # if the word is a negative word then turn on the flag
            if word in  self.neg_words:
                flag=1
            # if word is a punctuation then word off the flag
            if word in  self.punctuation:
                flag=0
                
        for word in not_stem:
            new_text=new_text+word+" "
        new_text=new_text[:-1]
        return new_text

 # valid alpabet or charactars
class CharVectorizer():
    def __init__(self, maxlen=1024, padding='pre', truncating='pre', alphabet="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|$%ˆ&*˜‘+=<>()[]{}#@"""):
        
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating

        self.char_dict = {'_pad_': 0, '_unk_': 1, ' ': 0} 
        for i, k in enumerate(self.alphabet, start=2):
            self.char_dict[k] = i

    def transform(self,sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        sequences = []

        for sentence in sentences:
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]
            
            if self.maxlen:
                length = len(seq)

                if length > self.maxlen:# we need to crop the sequence

                    if self.truncating == 'pre':# we crope from the end of the sequence
                        seq = seq[-self.maxlen:]
                    elif self.truncating == 'post':# we crop the beggining of the sequence
                        seq = seq[:self.maxlen]

                if length < self.maxlen:# we need to pad the sequence

                    diff = np.abs(length - self.maxlen)
                    if self.padding == 'pre':#We pad in the beggining
                        seq = [self.char_dict['_pad_']] * diff + seq
                    elif self.padding == 'post':#We pad at the end
                        seq = seq + [self.char_dict['_pad_']] * diff

            sequences.append(seq)                

        return sequences        
    
    def get_params(self):
        params = vars(self)
        return params
