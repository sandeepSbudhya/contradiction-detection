from transformers import AutoTokenizer
from huggingface_hub import from_pretrained_keras
import tensorflow as tf
import os.path

class BertMNLIModel:
    '''Class that loads pretrained bert model on the full mnli dataset for binary classification'''

    def __init__(self) -> None:
        '''Initialize the model and the tokenizer as well the exit map between the last layer of the model and output layer'''
        self.current_path = os.path.abspath(os.path.dirname(__file__))
        def loss():
            return
        self.model = from_pretrained_keras(os.path.join(self.current_path, 'models/fine_tuned_bert_on_mnli'), custom_objects={'dummy_loss':loss}, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.id2label = {0: "ENTAILMENT", 1:"NEUTRAL", 2: "CONTRADICTION"}
    
    def predict(self, sentance1, sentance2) -> str:
        '''
        function that accepts two sentances and checks if they contradict each other.
        A sentance contradicts another sentance if
        accepting one of them as True automatically implies that the other one is False
        
        Input
            sentance1: String
            sentance2: String

        Output
            String

        Sample Input: 
            "The car was yellow."
            "The car was blue."

        Posible predicted output: "CONTRADICTION"

        Sample Input: 
            "The sky is blue."
            "It was a rainy day yesterday."

        Possible predicted output: "NONCONTRADICTION"
        '''
        text = sentance1+' '+sentance2
        tokenized_text = self.tokenizer(text, return_tensors="tf")
        entry = {
            'attention_mask':tokenized_text['attention_mask'],
            'input_ids':tokenized_text['input_ids']
        }    
        logits = self.model(entry,)['logits']
        predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
        return self.id2label[predicted_class_id]

