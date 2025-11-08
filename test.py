from src.model.lm import HFModel

def test_hfmodel_generation():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = HFModel(model_name)
    prompt = "Hello, how are you?"
    generated_text = model.generate(prompt, max_length=20)
    print(f"Generated text: {generated_text}")

def test_hfmodel_gold_CE():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = HFModel(model_name)
    prompt = "My name is Emily. What is my Name?"
    gold = "Emily"
    nll, decoded_text, token_count = model.gold_CE(prompt, gold)
    print(f"NLL: {nll}, Decoded Text:{decoded_text}, Token Count: {token_count}")

if __name__ == "__main__":
    #test_hfmodel_generation()
    test_hfmodel_gold_CE()
   #test_hfmodel_generation()