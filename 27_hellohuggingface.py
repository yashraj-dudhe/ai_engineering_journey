from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def interact_with_gpt2():
    print("1.Downloading....")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    print("Model Downloaded")
    
    print("2.Preparing input")
    input_text = "The future of AI is"
    input_ids = tokenizer.encode(input_text,return_tensors = 'pt')
    
    print(f"input:{input_text}")
    print(f"ids: {input_ids}")
    
    print("3 Generation")
    
    output_ids = model.generate(
        input_ids,
        max_length = 30,
        num_return_sequences=1,
        do_sample = True,
        temperature = 0.7
    )
    
    generated_text = tokenizer.decode(output_ids[0],skip_special_tokens = True)
    print("Result")
    print(generated_text)
    

if __name__ == "__main__":
    interact_with_gpt2()