import torch
from src.model.lm import HFModel
import datasets
import src.prompts as prompts
import os
from tqdm import tqdm # For a nice progress bar

from pprint import pprint
class Scorer:

    def __init__(self, model: HFModel, dataset: datasets.Dataset, batch_size: int = 32):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_token = 200

    def _prepare_prompts(self, batch):
        """
        CPU-safe function to prepare prompts. This will be run in parallel.
        It does NOT touch the model or GPU.
        """
        system_prompt = prompts.GSM8K
        questions = batch["question"]
        
        # Prepare conversations for the batch
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ] for question in questions
        ]
        
        # This is just text manipulation, perfect for multiprocessing
        return {"conversations": conversations}

    def score(self):
        """
        Scores the dataset by first preparing prompts on multiple CPUs,
        then running inference on the GPU in a single process.
        """
        # --- 1. CPU-Bound Pre-processing ---
        num_cpus = os.cpu_count() or 1
        num_workers = min(num_cpus, 64)
        print(f"Preparing prompts using {num_workers} CPU workers...")

        # Use .map() to prepare all conversations in parallel
        prepared_dataset = self.dataset.map(
            self._prepare_prompts,
            batched=True,
            batch_size=self.batch_size,
            num_proc=num_workers,
            desc="Preparing Prompts"
        )

        # --- 2. GPU-Bound Inference ---
        print("\nStarting GPU inference...")
        generated_responses = []
        
        # Iterate over the prepared dataset in batches in the main process
        for i in tqdm(range(0, len(prepared_dataset), self.batch_size), desc="Generating Responses"):
            batch = prepared_dataset[i:i+self.batch_size]
            
            # The batch contains the 'conversations' column we just made
            conversations_batch = batch["conversations"]
            
            # Now call the GPU model with the prepared batch
            responses = self.model.chat(conversations_batch, max_new_tokens=self.max_token)
            generated_responses.extend(responses)
            
        # --- 3. Add results back to the dataset ---
        final_dataset = prepared_dataset.add_column("Generated Response", generated_responses)

        print("\n--- Scoring Complete ---")
        pprint(final_dataset[0])
        return final_dataset