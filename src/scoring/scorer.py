import torch
from src.model.lm import HFModel
import datasets
import src.prompts as prompts
import os
from multiprocessing import set_start_method
import re  
from src.utils.parser import Parser

class Scorer:

    def __init__(self, model: HFModel,
                 dataset: datasets.Dataset,
                 batch_size: int,
                 responses_per_sample: int, 
                 max_new_token: int, 
                 temperature: float,
                 enable_thinking: bool):

        self.model = model
        self.dataset = dataset
        self.max_new_token = max_new_token
        
        if batch_size <= 0 or batch_size > len(dataset):
            raise ValueError(f"Invalid batch size: {batch_size}")
        
        self.batch_size = batch_size
        self.responses_per_samples = responses_per_sample
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        
    def score(self):
        
        """
        Score entire dataset in batches, calculate accuracy, and save the resulting dataset.
        """
    
        print(f"Scoring dataset {self.dataset.info.dataset_name} with {len(self.dataset)} samples using batch size {self.batch_size}...")
        
   
        result_dataset = self.dataset.map(
            self._score_batch,
            batched=True,
            batch_size=self.batch_size,
            desc="Scoring batches", 
        )
        
        print("Calculating accuracy for scored dataset...")
        result_dataset = result_dataset.map(
            self._calculate_accuracy,
            desc="Calculating accuracy"
        )
        
        save_path = os.path.join("results", f"{self.dataset.info.dataset_name}_scored")
        print(f"Saving final dataset to {save_path}...")
        result_dataset.save_to_disk(save_path)


    def _score_batch(self, batch):

        system_prompt = prompts.GSM8K
        questions = batch["question"]
        
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ] for question in questions
        ]
        
        output_data = {}
        # we generate 20 responses for each question (range(20) -> 0 to 19)
        for i in range(self.responses_per_samples):
            
            responses = self.model.chat(
                conversations, 
                max_new_tokens=self.max_new_token, 
                temperature=self.temperature,
                enable_thinking=self.enable_thinking
            )
            
            output_data[f"response_{i}"] = responses 

        return output_data

    def _calculate_accuracy(self, example):
        """
        This function is applied to a single example (row) from the dataset.
        It automatically detects all 'response_X' columns, compares them
        to the true answer numerically (if possible), and returns the accuracy.
        (Private method)
        """
       
        response_keys = [key for key in example.keys() if re.match(r'^response_\d+$', key)]
        num_responses = len(response_keys)
        
        if num_responses == 0:
            return {"accuracy": 0.0} 


        true_num = float(example["answer_num"])
        print("true_num ", true_num)
        
        correct_count = 0
        for response_key in response_keys:
            
            generated_response_str = str(example[response_key]).strip()
            
            gen_num = Parser.extract_generated_number(generated_response_str)
            print(f"Generated response for {response_key}: {generated_response_str} -> Parsed number: {gen_num}")
  
            
            if gen_num is not None:
                if true_num == gen_num:
                    correct_count += 1
            
        accuracy = correct_count / num_responses
        return {"accuracy": accuracy} 