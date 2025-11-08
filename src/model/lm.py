from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFModel:  
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 50) -> str:
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(**inputs, max_length=max_length) #TODO: add other generation parameters as needed
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def gold_CE(self, prompt: str, gold: str) -> tuple[float, str, int]: #TODO: make this work for batched inputs
        
        device = next(self.model.parameters()).device
        
        x = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        y = self.tokenizer(gold, add_special_tokens=True, return_tensors="pt")

        # Debug prints to verify sizes
      
        input_ids = torch.cat([x["input_ids"], y["input_ids"]], dim=1)  # concatenate prompt and gold
        ignore = torch.full_like(x["input_ids"], -100)  # ignore prompt tokens in loss
        labels = torch.cat([ignore, y["input_ids"]], dim=1)
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = torch.cat([x["attention_mask"], y["attention_mask"]], dim=1).to(device)
        
        out = self.model(input_ids, labels=labels, attention_mask=attention_mask)  # contains logits and loss
        predicted_ids = torch.argmax(out.logits, dim=-1)
        
        decoded_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        #remove last logit to align predictions with target for next token prediction
        logits = out.logits[:, :-1, :]   # [batch, seq_len-1, vocab_size] 

        #remove first token from target to align with shifted logits
        labels = labels[:, 1:] # [batch, seq_len-1]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index = -100) #no reduction to keep per-token loss
        
        #logits are reshaped to [batch*seq_len, vocab_size] and labels to [batch*seq_len]
        per_tok = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)) # [batch*seq_len]
        
        per_tok = per_tok.view(labels.size())    # [batch, seq_len-1]
     
        gold_mask = (labels != -100).float()  #mask to consider only gold tokens
        tok_sum = (per_tok * gold_mask).sum(dim=1)
        
        tok_cnt = gold_mask.sum(dim=1).clamp_min(1) #count tokens, prevent div by 0
        
        nll = (tok_sum / tok_cnt).item() #average NLL per token
        
        return nll, decoded_text.strip(), int(tok_cnt.item())