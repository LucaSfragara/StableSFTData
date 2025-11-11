from re import A
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from typing import List, Dict
class HFModel:  
    def __init__(self, model_name: str):

        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            device_map="auto", 
            attn_implementation="sdpa"
        ).eval()
        print(f"Model device: {self.model.device}")

        if self.model.device == "cpu":
            raise ValueError("Model is on CPU. Please ensure you have a compatible GPU and the necessary libraries installed.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        """if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        """
    
    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        #get tokens generate per second
        
        #start_time = torch.cuda.Event(enable_timing=True)
        
        #start_time.record()
        outputs = self.model.generate(**inputs, max_length=max_length) #TODO: add other generation parameters as needed
        """
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
        """
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def chat(self, conversations: List[List[Dict[str, str]]], max_new_tokens: int) -> List[str]:
        """
        Generates responses for a batch of conversations in parallel.

        Args:
            conversations: A list of conversations. Each conversation is a list of
                         message dictionaries, e.g.,
                         [
                             [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "What is 2+2?"}],
                             [{"role": "system", "content": "Be a pirate."}, {"role": "user", "content": "How are you?"}]
                         ]
            max_new_tokens: The maximum number of new tokens to generate for each response.

        Returns:
            A list of generated response strings.
        """
        torch.cuda.empty_cache()
        gc.collect()
     
        prompts = [
            self.tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False, #enable_thinking = True
            ) for conv in conversations
        ]
        batch_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        #start_time = torch.cuda.Event(enable_timing=True)
        #start_time.record()
        
        out = self.model.generate(
            **batch_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            use_cache=True, 
            temperature=1, 
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
      
        """
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
        """
        
        input_token_len = batch_inputs.input_ids.shape[1] #length of input prompt tokens
        generated_ids = out[:, input_token_len:]  #get only generated tokens
    
        decoded_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return [text.strip() for text in decoded_texts]

    @torch.no_grad()
    def gold_CE(self, prompt: str, gold: str) -> tuple[float, str, int]: #TODO: make this work for batched inputs
        
        """
        Compute the average cross-entropy (negative log-likelihood) of a gold continuation given a prompt.
        This method tokenizes the prompt and gold texts, concatenates them as a single input sequence,
        and computes next-token cross-entropy while ignoring loss on the prompt tokens. It also returns
        a greedy-decoded sequence from the model's logits for inspection.
        Args:
            prompt: The input prompt text.
            gold: The gold/target continuation text whose tokens will contribute to the loss.
        Returns:
            Tuple[float, str, int]:
                - nll: The average per-token negative log-likelihood over the gold tokens only.
                - decoded_text: The greedy-decoded text from the model logits for the full sequence
                  (prompt + gold), with special tokens removed.
                - token_count: The number of gold tokens that contributed to the loss (i.e., non-ignored tokens).
        Notes:
            - Prompt tokens are assigned an ignore index (-100) and do not contribute to the loss.
            - Loss is computed using next-token prediction by shifting logits/labels.
            - Currently supports only batch size 1.
        """
        
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