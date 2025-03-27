import json

def process_jsonl(input_path, output_path):
    # 创建新的JSONL文件
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            # 遍历conversations数组
            for conv in data.get('conversations', []):
                # 统一value字段类型为字符串
                if isinstance(conv['value'], (list, dict)):
                    conv['value'] = json.dumps(conv['value'])  # 序列化数组/字典
            # 写回处理后的数据
            fout.write(json.dumps(data) + '\n')

# 使用示例
process_jsonl('/home/tc_workspace/code/VideoLLaMA3/data/child_llama3_post_test1.jsonl', '/home/tc_workspace/code/VideoLLaMA3/data/child_llama3_post_test2.jsonl')