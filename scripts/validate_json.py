import json
import pyarrow.json as paj

def validate_jsonl(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ 第 {i} 行错误: {e}")
                print("问题行内容:", line.strip())
                if i == 1000:
                    return
            except Exception as e:
                print(f"⚠️ 第 {i} 行异常: {str(e)}")

# validate_jsonl('/data0/tc_workspace/internlm/code/VideoLLaMA3/data/child_llama3.jsonl')

import json
from jsonschema import validate

import json
import re

def fix_jsonl(input_path, output_path):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            # 修复键名引号问题
            fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', line)
            # 转义双引号
            fixed = fixed.replace('"', '\\"')
            try:
                json.loads(fixed)  # 二次验证
                fout.write(fixed + '\n')
            except:
                continue

fix_jsonl('/data0/tc_workspace/internlm/code/VideoLLaMA3/data/child_llama3_test.jsonl', '/data0/tc_workspace/internlm/code/VideoLLaMA3/data/child_llama3_test_fixed.jsonl')