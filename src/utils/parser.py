import re

class Parser:
    
    @staticmethod
    def extract_generated_number(text: str) -> float | None:
        
        # remove <think>...</think>, then take the last number
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', clean)
        return float(nums[-1]) if nums else None

    @staticmethod
    def extract_true_number(text: str) -> float:
        
        tags = re.findall(r"####\s*([^\n\r]+)", text)
    
        if tags:
            return float(tags[-1].strip().strip(".").replace(",", "").strip())
        
        
        
if __name__ == "__main__":
    
    sample_generated_text = """
        "<think>
        Okay, let's see. Natalia sold clips to 48 friends in April. Then she sold half as many in May. So first, I need to find out how many clips she sold in May. Half as many as April's 48. So half of 48 is... let me calculate that. 48 divided by 2 is 24. So she sold 24 clips in May. Now, to find the total for April and May, I add 48 (April) and 24 (May). Adding those together: 48 + 24 equals 72. So the answer should be 72 clips. Let me check again. April is 48, May is half, so 48 divided by 2 is 24. 48 + 24 is indeed 72. Yep, that seems right.
        </think>
        Answer: 72$ "
    """

    sample_true_text = "The total clips sold is #### 72."
    extracted = Parser.extract_generated_number(sample_generated_text)
    extracted_true = Parser.extract_true_number(sample_true_text)
    print(f"Extracted number: {extracted}")
    print(f"Extracted true number: {extracted_true}")