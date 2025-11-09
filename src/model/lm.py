from re import A
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
class HFModel:  
    def __init__(self, model_name: str):

        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            device_map="auto", 
            attn_implementation="sdpa"

        )
        print(f"Model device: {self.model.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
    
    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 50) -> str:
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        #get tokens generate per second
        
        start_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        outputs = self.model.generate(**inputs, max_length=max_length) #TODO: add other generation parameters as needed
        
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # convert to
        end_tokens = outputs.shape[1]
        tokens_generated = end_tokens 
        tokens_per_second = tokens_generated / elapsed_time
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nGeneration Stats:")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def chat(self, user: str, system: str, max_new_tokens: int) -> str:
        
        torch.cuda.empty_cache()
        gc.collect()
        
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking = False
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )

        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()

        elapsed_time = start_time.elapsed_time(end_time) / 1000  # convert to
        end_tokens = out.shape[1]
        tokens_generated = end_tokens 
        tokens_per_second = tokens_generated / elapsed_time
                
        print(f"\nGeneration Stats:")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        
        
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

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