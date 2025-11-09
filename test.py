from src.model.lm import HFModel
from src.data.loader import load_gsm8k_dataset

def test_hfmodel_generation():
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    dataset = load_gsm8k_dataset("train")
    prompt = dataset[0].question
    print(f"Prompt: {prompt}")
    generated_text = model.chat(user=prompt, system="Solve this high school math problem.", max_new_tokens=200)
    print(f"Generated text: {generated_text}")

def test_hfmodel_gold_CE():
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    prompt = "Capital of France is"
    gold = "Paris"
    nll, decoded_text, token_count = model.gold_CE(prompt, gold)
    print(f"NLL: {nll}, Decoded Text:{decoded_text}, Token Count: {token_count}")

if __name__ == "__main__":

    #print("Testing HFModel...")
    test_hfmodel_generation()
    #test_hfmodel_gold_CE()
   #test_hfmodel_generation()