from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/voxlinux_models/t5_metasploit_base"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

def msf_nl2cmd(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print(msf_nl2cmd("scan all open ports on 192.168.1.10"))
