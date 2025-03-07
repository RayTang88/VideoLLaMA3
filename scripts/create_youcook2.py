import xml.etree.ElementTree as ET
import json
import random
import os

from tqdm import tqdm

def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    # print(merged, sorted_intervals)
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    return merged

def parse_annotation(eaf_file):
    xml_content = open(eaf_file, 'r', encoding='utf-8').read()
    root = ET.fromstring(xml_content)
    
    # 提取媒体信息
    media_desc = root.find(".//MEDIA_DESCRIPTOR")
    video_id = media_desc.get("RELATIVE_MEDIA_URL").split('/')[-1].split('.')[0]
    video_path = eaf_file.replace("annotations", "videos").replace(".eaf", ".mp4")
    
    # 构建基础数据
    base_data = {
        "video_url": media_desc.get("MEDIA_URL"),
        "video_path": video_path,
        "youtube_id": video_id
    }
    
    # 解析时间槽
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        time_slots[ts.get("TIME_SLOT_ID")] = int(ts.get("TIME_VALUE"))
    
    # 处理标注数据
    annotations = []
    used_intervals = []
    
    for idx, ann in enumerate(root.findall(".//TIER[@TIER_ID='cls']/ANNOTATION")):
        align = ann.find("ALIGNABLE_ANNOTATION")
        ref1, ref2 = align.get("TIME_SLOT_REF1"), align.get("TIME_SLOT_REF2")
        start, end = time_slots[ref1], time_slots[ref2]
        recipe_type = align.find("ANNOTATION_VALUE").text
        
        annotations.append({
            "id": f"{video_id}_{idx}",
            **base_data,
            "recipe_type": recipe_type,
            "segment": [start, end],
            "sentence": recipe_type
        })
        used_intervals.append((start, end))
    
    # 处理未标注时间段
    merged = merge_intervals(used_intervals)
    all_times = sorted(time_slots.values())
    total_range = [all_times[0], all_times[-1]]
    
    gaps = []
    prev_end = total_range[0]
    
    ggap = 200
    for interval in merged:
        if interval[0] > prev_end and ((interval[0]-prev_end) > ggap):
            #这里增加一个前后缩25帧的策略
            gaps.append([prev_end + int(ggap/8), interval[0]+ int(ggap/8)])
            
        prev_end = max(prev_end, interval[1])
    
    if gaps:
        selected = random.choice(gaps)
        annotations.append({
            "id": f"{video_id}_other",
            **base_data,
            "recipe_type": "other",
            "segment": selected,
            "sentence": "other"
        })
    
    # return '\n'.join(json.dumps(ann) for ann in annotations)
    return annotations

def save_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 序列化时保留非ASCII字符[4](@ref)
            json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
            f.write(json_line + '\n')  # 按JSONL规范添加换行符[3](@ref)
            # f.write(json.dumps(item, indent=2, ensure_ascii=False))

def find_eaf_files(root_dir):
    """
    递归查找指定目录下所有.eaf文件
    :param root_dir: 要搜索的根目录路径
    :return: 包含完整路径的.eaf文件列表
    """
    eaf_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.eaf'):
                full_path = os.path.join(dirpath, filename)
                eaf_files.append(full_path)
    return eaf_files 

def split_dataset(data):
    """
    将数据集按5%验证集、5%测试集、90%训练集随机分割
    参数：
        data: 输入数据列表
    返回：
        (训练集, 验证集, 测试集) 三元组
    """
    n = len(data)
    indices = list(range(n))
    
    # 打乱索引顺序[8](@ref)
    random.shuffle(indices)
    
    # 计算分割点（至少保留1个样本）[7](@ref)
    val_size = max(1, round(n * 0.05))
    test_size = max(1, round(n * 0.05))
    
    # 防止超出索引范围
    split_point = min(val_size + test_size, n)
    
    # 分割索引
    val_indices = indices[:val_size]
    test_indices = indices[val_size:split_point]
    train_indices = indices[split_point:]
    
    # 按索引提取数据
    return (
        [data[i] for i in train_indices],
        [data[i] for i in val_indices],
        [data[i] for i in test_indices]
    ) 
    
def save_to_jsonl(results, jsonl_name):
    train_data, val_data, test_data =split_dataset(results)
    
    save_jsonl(train_data, jsonl_name.replace(".jsonl", "_train.jsonl"))
    save_jsonl(val_data, jsonl_name.replace(".jsonl", "_val.jsonl"))
    save_jsonl(test_data, jsonl_name.replace(".jsonl", "_test.jsonl"))

if __name__ == "__main__":
    
    
    root_dir = "/home/ubuntu/PublicData/children_actions/annotations/"
    
    eaf_lists = find_eaf_files(root_dir)
    
    results = []
    for eaf_file in tqdm(eaf_lists):
        result = parse_annotation(eaf_file)
        results.extend(result)

    # 生成JSONL文件
    save_to_jsonl(results, '/data0/tc_workspace/internlm/code/VideoLLaMA3/data/child_llama3.jsonl')
    
    # print(result)