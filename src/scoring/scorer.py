import torch
from src.model.lm import HFModel
import datasets
import src.prompts as prompts
import os
from multiprocessing import set_start_method

from pprint import pprint
class Scorer:

    def __init__(self, model: HFModel, dataset: datasets.Dataset, batch_size: int = 32):
        
        self.model = model
        self.dataset = dataset
        
        self.max_token = 100
        
        if batch_size <= 0 or batch_size > len(dataset):
            raise ValueError(f"Invalid batch size: {batch_size}")
        self.batch_size = batch_size

    def score(self):
        
        """
        Score entire dataset in batches and return list of ScoreRecords
        """
        print(f"Scoring dataset {self.dataset.info.dataset_name} with {len(self.dataset)} samples using batch size {self.batch_size}...")
        

        result_dataset = self.dataset.map(
            self._score_batch,
            batched=True,
            batch_size=self.batch_size,
            desc="Scoring batches", 
            num_proc=1
        )
        #save dataset 
        result_dataset.save_to_disk(os.path.join("results", f"{self.dataset.info.dataset_name}_scored"))

    def _score_batch(self, batch):

        system_prompt = prompts.GSM8K
        
        questions = batch["question"]
        
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ] for question in questions
        ]
        
        responses = self.model.chat(conversations, max_new_tokens=200)
        
        return {"Generated Response": responses}