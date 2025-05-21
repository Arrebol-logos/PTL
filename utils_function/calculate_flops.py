from calflops import calculate_flops
from transformers import AutoModel
from transformers import AutoTokenizer

batch_size, max_seq_length = 1, 128
model_name = "meta-llama/Meta-Llama-3-8B"
model_save = model_name
model = AutoModel.from_pretrained(model_save)
tokenizer = AutoTokenizer.from_pretrained(model_save)

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size,max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("Result FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))