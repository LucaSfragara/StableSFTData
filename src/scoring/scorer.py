import torch
from src.model.lm import HFModel
import datasets
import src.prompts as prompts
import os
from multiprocessing import set_start_method
import re  

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
        Score entire dataset in batches, calculate accuracy, and save the resulting dataset.
        """
        #### TO REMOVE: For testing, we will only score a subset of the dataset
        dataset_to_score = self.dataset.select(range(512))
        print(f"Scoring dataset {self.dataset.info.dataset_name} with {len(dataset_to_score)} samples using batch size {self.batch_size}...")
        
        #print(f"Scoring dataset {self.dataset.info.dataset_name} with {len(self.dataset)} samples using batch size {self.batch_size}...")
        ### TO CHANGE NAME OF DATASET
        
        result_dataset = dataset_to_score.map(
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
        for i in range(20):
            
            responses = self.model.chat(
                conversations, 
                max_new_tokens=200
            )
            
            output_data[f"response_{i}"] = responses 

        return output_data


    def _clean_and_convert_to_float(self, s):
        """
        Attempts to clean a string and convert it to a float.
        Removes common non-numeric characters like $, ,, %, etc.
        (Private method)
        """
        if not isinstance(s, str):
            s = str(s)

        cleaned_s = re.sub(r'[^\d.-]', '', s)
        
        try:
            return float(cleaned_s)
        except (ValueError, TypeError):
            return None

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

        try:
            true_answer_str = example['answer'].split('####')[-1].strip()
        except Exception as e:
            return {"accuracy": 0.0} 

        true_num = self._clean_and_convert_to_float(true_answer_str) 

        correct_count = 0
        for response_key in response_keys:
            
            generated_response_str = str(example[response_key]).strip()
            
            gen_num = self._clean_and_convert_to_float(generated_response_str) # <-- Call internal method

            is_correct = False
            
            if true_num is not None and gen_num is not None:
                if true_num == gen_num:
                    is_correct = True
            else: 
                if true_answer_str == generated_response_str:
                    is_correct = True
            
            if is_correct:
                correct_count += 1
                
        accuracy = correct_count / num_responses
        return {"accuracy": accuracy} 