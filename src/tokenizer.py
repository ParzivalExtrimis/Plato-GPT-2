import nltk
from nltk.tokenize import word_tokenize
import string
from typing import Literal, List, Tuple, Dict

class Tokenizer:
    def __init__(self, tokenizer_type: Literal['char', 'word', 'nltk']) -> None:
        self._tokenizer_type = tokenizer_type
        self._tokens = None
        self._size = None
        self._vocab = None
        self._vocab_size = None

        self._stoi = None
        self._itos = None

    def get_size(self) -> int:
        assert self._size is not None, 'Something went wrong in initializing the tokenizer. Dataset-Size is None'
        return self._size
    
    def get_vocab_size(self) -> int:
        assert self._vocab_size is not None, 'Something went wrong in initializing the tokenizer. Vocab-Size is None'
        return self._vocab_size

    def get_maps(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        self._set_attribs(text)
        assert self._vocab is not None, "Something went wrong in initializing the tokenizer. Vocab is None"

        self._stoi = { tok:i for i,tok in enumerate(self._vocab) }
        self._itos = { i:tok for i,tok in enumerate(self._vocab) }

        return self._stoi, self._itos
        
    def encode(self) -> List[int]:
        assert self._stoi is not None, 'Something went wrong in initializing the tokenizer. STOI Map is None. \n Please ensure `get_maps()` has been called on dataset.'
        return [self._stoi[tok] for tok in self._tokens]
    
    def decode(self, int_arr: List[int]) -> str:
        assert self._itos is not None, 'Something went wrong in initializing the tokenizer. ITOS Map is None. \n Please ensure `get_maps()` has been called on dataset.'
        decoded = [self._itos[i] for i in int_arr]
        return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in decoded]).strip()
    


    # internal initializer funcs

    def _char_lv_tokenize(self, text: str) -> None:
        # here are all the unique characters that occur in this text
        self._tokens = list(text)
        self._size = len(self._tokens)
        self._vocab = sorted(list(set(text)))
        self._vocab_size = len(self._vocab)
        
    def _word_lv_tokenize(self, text: str) -> None:
        self._tokens = text.split(' ')
        self._size = len(self._tokens)
        self._vocab = sorted(list(set(self._tokens)))
        self._vocab_size = len(self._vocab)

    def _nltk_tokenize(self, text: str) -> None:
        nltk.download('punkt')
        self._tokens = word_tokenize(text=text)
        self._size = len(self._tokens)
        self._vocab = sorted(list(set(self._tokens)))
        self._vocab_size = len(self._vocab)

    #switch tokens based on tokenizer selected
    def _set_attribs(self, text: str) -> Tuple[List[str], int, int]:
        if self._tokenizer_type == 'char':
            self._char_lv_tokenize(text)
        elif self._tokenizer_type == 'word':
            self._word_lv_tokenize(text)
        elif self._tokenizer_type == 'nltk':
            self._nltk_tokenize(text)
        else:
            print('Invalid token type.')
            raise ValueError('No such tokenizer defined. Use a valid tokenizer type. [ char | word | nltk ]') 
