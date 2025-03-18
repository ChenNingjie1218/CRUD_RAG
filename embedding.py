from transformers import AutoModel, AutoTokenizer

# 模型名称
model_name = "BAAI/bge-base-zh-v1.5"

# 本地保存路径
save_directory = "src/sentence-transformers/bge-base-zh-v1.5"

# 下载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到本地
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"模型已保存到：{save_directory}")