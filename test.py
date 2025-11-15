from src.model.lm import HFModel
from src.data.loader import load_gsm8k_dataset
from src.core.types import Dataset
from src.scoring.scorer import Scorer
import datasets
from datasets.config import HF_DATASETS_CACHE
import os
from src.prompts import GSM8K_FINE_TUNE, GSM8K

def test_hfmodel_generation():
    #model_name = "Qwen/Qwen3-0.6B"
    model_name = "allenai/open-instruct-pythia-6.9b-tulu"
    
    model = HFModel(model_name)
    dataset = load_gsm8k_dataset("train")
    prompt = dataset[0].question
    #print(f"Prompt: {prompt}")
    
    conversations_batch = [
        # Conversation 1
        [
            {"role": "system", "content": GSM8K},
            {"role": "user", "content": "Benny threw bologna at his balloons.  He threw two pieces of bologna at each red balloon and three pieces of bologna at each yellow balloon.  If Benny threw 58 pieces of bologna at a bundle of red and yellow balloons, and twenty of the balloons were red, then how many of the balloons in the bundle were yellow?"}
        ]
    ]
    
    generated_text = model.chat(conversations_batch,
                                max_new_tokens=400,
                                temperature=0, 
                                use_custom_chat_template=True)
    
    #generated_text = model.generate(
    #    prompt="""
    #            <|user|>
    #            You are a math assistant solving simple math question. RRespond to the question by thinking step by step. Question: 
    #            Natalia sold 48 clips in April and half as many in May. How many clips did she sell in total?,
    #            <|assistant|> \n
    #            """,
    #    max_length=500)
    
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
    #dataset = datasets.load_dataset("gsm8k_processed")['train']
    dataset = datasets.load_from_disk(os.path.join(HF_DATASETS_CACHE, "gsm8k_processed"))['train']
   # dataset_d = Dataset.from_list([sample.__dict__ for sample in dataset])

    scorer = Scorer(model,
                    dataset, #type: ignore
                    batch_size=64, 
                    responses_per_sample=5,
                    max_new_token=512,
                    temperature=0.2,
                    enable_thinking=False)
    scorer.score()

if __name__ == "__main__":

    #print("Testing HFModel...")
    test_hfmodel_generation()
    #score()
    
    #test_hfmodel_gold_CE()
   #test_hfmodel_generation()