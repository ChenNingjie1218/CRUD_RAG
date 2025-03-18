from transformers import AutoModel, AutoTokenizer

# 模型名称
model_name = "shibing624/text2vec-base-chinese"

# 本地保存路径
save_directory = "src/.cache/text2vec-base-chinese"

# 下载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到本地
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"模型已保存到：{save_directory}")