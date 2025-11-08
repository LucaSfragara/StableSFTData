from src.model.lm import HFModel

def test_hfmodel_generation():
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    prompt = "Capital of France is"
    #generated_text = model.generate(prompt, max_length=20)
    generated_text = model.chat(user=prompt, system="You are a helpful assistant.", max_new_tokens=100)
    print(f"Generated text: {generated_text}")

def test_hfmodel_gold_CE():
    model_name = "Qwen/Qwen3-0.6B"
    model = HFModel(model_name)
    prompt = "Capital of France is"
    gold = "Paris"
    nll, decoded_text, token_count = model.gold_CE(prompt, gold)
    print(f"NLL: {nll}, Decoded Text:{decoded_text}, Token Count: {token_count}")

if __name__ == "__main__":
    test_hfmodel_generation()
    #test_hfmodel_gold_CE()
   #test_hfmodel_generation()