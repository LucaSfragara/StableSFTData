from src.model.lm import HFModel
from src.data.loader import load_gsm8k_dataset
from src.core.types import Dataset
from src.scoring.scorer import Scorer
import datasets

def test_hfmodel_generation():
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    dataset = load_gsm8k_dataset("train")
    prompt = dataset[0].question
    #print(f"Prompt: {prompt}")
    
    conversations_batch = [
        # Conversation 1
        [
            {"role": "system", "content": "You are a math expert."},
            {"role": "user", "content": "Natalia sold 48 clips in April and half as many in May. How many clips did she sell in total?"}
        ],
        """# Conversation 2
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        # Conversation 3
        [
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a single sentence about a robot who discovers music."}
        ]"""
    ]
    
    generated_text = model.chat(conversations_batch, max_new_tokens=200)
    #generated_text = model.generate(prompt="The capital of France is", max_length=20)
    print(f"Generated text: {generated_text}") 

def test_hfmodel_gold_CE():
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    prompt = "Capital of France is"
    gold = "Paris"
    nll, decoded_text, token_count = model.gold_CE(prompt, gold)
    print(f"NLL: {nll}, Decoded Text:{decoded_text}, Token Count: {token_count}")


def score():
    
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    dataset = datasets.load_dataset("gsm8k", 'main')['train']
   # dataset_d = Dataset.from_list([sample.__dict__ for sample in dataset])

    scorer = Scorer(model, dataset, batch_size=512)
    scorer.score()

if __name__ == "__main__":

    #print("Testing HFModel...")
    #test_hfmodel_generation()
    score()
    #test_hfmodel_gold_CE()
   #test_hfmodel_generation()d
   