import torch
from torch.utils.data import DataLoader, Dataset


class LMDataset(Dataset):
    def __init__(self, text, tokenizer, context_length=1024, stride=128):
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids)-context_length, stride):
            self.input_ids.append(token_ids[i:i+context_length])
            self.target_ids.append(token_ids[i+1:i+context_length+1])
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.target_ids[index])
    

if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    text = open("/home/aki/workspace/learning/finetuning/data/shakespeare_sample.txt").read()
    dataset = LMDataset(text, tokenizer)
    print(dataset[0])