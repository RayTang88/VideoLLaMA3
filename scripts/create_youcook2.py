import xml.etree.ElementTree as ET
import json
import random
import os
import math

from tqdm import tqdm
from run_qwen72b_vlm import VLLM


calib_dict = {'cry':'cry', 'laugh':'laugh', 'eat_hands':'put hands in mouth', 'feed':'being fed', 'eat':'eat', 'drink': 'drink', 'lie':'lie down', 'fall_backward':'fall_backward', 'crawl':'crawl', 'sit':'sit', 'on_feet':'stand', 'walk':'walk', 'dance':'dance', 'jump':'jump', 'run':'run'
}

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
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in data:
            # 序列化时保留非ASCII字符[4](@ref)
            json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
            f.write(json_line + '\n')  # 按JSONL规范添加换行符[3](@ref)
            # f.write(json.dumps(item, indent=2, ensure_ascii=False))

def get_existing_ids(train_json, val_json):
    existing_ids = []
    for json_path in [train_json, val_json]:
        with open(json_path, 'r', encoding='utf-8') as fin:
            # 逐行解析JSONL
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    # 检查ID字段是否存在
                    if 'id' in data and isinstance(data['id'], str):
                        current_id = data['id']
                        existing_ids.append(current_id)
                    
                except json.JSONDecodeError:
                    print(f"无效JSON格式: {line[:50]}...")
    return existing_ids                
            
def find_eaf_files(root_dir):
    """
    递归查找指定目录下所有.eaf文件
    :param root_dir: 要搜索的根目录路径
    :return: 包含完整路径的.eaf文件列表
    """
    train_json = '/home/tc_workspace/code/VideoLLaMA3/data/child_llama3_post_train.jsonl'
    test_json = '/home/tc_workspace/code/VideoLLaMA3/data/child_llama3_post_test.jsonl'
    existing_ids = get_existing_ids(train_json, test_json)
    eaf_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.eaf'):
                if not filename[:-4] in existing_ids:
                    full_path = os.path.join(dirpath, filename)
                    eaf_files.append(full_path)
                    existing_ids.append(filename[:-4])
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
    # val_size = max(1, round(n * 0.05))
    test_size = max(1, round(n * 0.01))
    
    # 防止超出索引范围
    # split_point = min(val_size + test_size, n)
    split_point = min(test_size, n)
    
    # 分割索引
    # val_indices = indices[:val_size]
    test_indices = indices[:split_point]
    train_indices = indices[split_point:]
    
    # 按索引提取数据
    return (
        [data[i] for i in train_indices],
        # [data[i] for i in val_indices],
        [data[i] for i in test_indices]
    ) 
    
def save_to_jsonl(results, jsonl_name):
    train_data, test_data =split_dataset(results)
    
    save_jsonl(train_data, jsonl_name.replace(".jsonl", "_post_train.jsonl"))
    # save_jsonl(val_data, jsonl_name.replace(".jsonl", "_pre_val.jsonl"))
    save_jsonl(test_data, jsonl_name.replace(".jsonl", "_post_test.jsonl"))
    
def covert_time(time_slots, ref1):

    seconds = float(time_slots[ref1]/1000)
    mins = math.floor(seconds / 60)
    secs = (seconds % 60).toFixed(1)
    return ""


def convert_milliseconds(time_slots, ref1):
    seconds = time_slots[ref1]
    total_seconds = seconds // 1000         # 总秒数
    milliseconds = seconds % 1000           # 剩余毫秒
    
    hours = (total_seconds // 3600) % 24

    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# 测试5329ms转换
# print(convert_milliseconds(5329))  # 输出: 00:00:05.329

    


def get_total(eaf_file, vllm_classs):
    xml_content = open(eaf_file, 'r', encoding='utf-8').read()
    root = ET.fromstring(xml_content)
    
    # 提取媒体信息
    video_path = eaf_file.replace("/home/tc_workspace/code/VideoLLaMA3/data/annotations", "/mnt/tos-tommi-algorithm/tc_workspace/data/videos").replace(".eaf", ".mp4")
    if not os.path.exists(video_path):
        print("----",video_path)
        return None
    
    media_desc = root.find(".//MEDIA_DESCRIPTOR")
    video_id = media_desc.get("RELATIVE_MEDIA_URL").split('/')[-1].split('.')[0]
    # 构建基础数据
    base_data = {
        "id": video_id,
        "video": [video_path]
    }
    
    # 解析时间槽
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        time_slots[ts.get("TIME_SLOT_ID")] = int(ts.get("TIME_VALUE"))
    
    # 处理标注数据
    annotations = []
    recipe_types = []
    segments = []
    values =[]
    for idx, ann in enumerate(root.findall(".//TIER[@TIER_ID='cls']/ANNOTATION")):
        value = {}
        align = ann.find("ALIGNABLE_ANNOTATION")
        ref1, ref2 = align.get("TIME_SLOT_REF1"), align.get("TIME_SLOT_REF2")
        # start_f, end_f = float(time_slots[ref1]/1000), float(time_slots[ref2]/1000)
        start = convert_milliseconds(time_slots, ref1)
        end = convert_milliseconds(time_slots, ref2)
        recipe_type = align.find("ANNOTATION_VALUE").text
        recipe_type = calib_dict[recipe_type] if recipe_type in calib_dict else recipe_type
        
        recipe_types.append(recipe_type)
        segments.append([start, end])
        
        # values+="%.3f-%.3fs, %s;"%(start, end, recipe_type)
        value["behavior"] = recipe_type
        value["start"] = start
        value["end"] = end
        values.append(value)
    # values = f"[{', '.join(map(str, values))}]"    
    good_json = vllm_classs.infer(video_path=video_path)
    conversations = good_json[0]["conversations"]
    
    behavior_list = [
        {
            "from": "human",
            "value": "Observe the child in the video and identify if any of the following behaviors occur: ['cry', 'laugh', 'put hands in mouth', 'being fed', 'eat', 'drink', 'lie down', 'fall backward', 'crawl', 'sit', 'stand', 'walk', 'dance', 'jump', 'run']. For each observed behavior, provide the behavior name along with its corresponding start and end timestamps in JSON format."
        },
        {
            "from": "gpt",
            "value": str(values)
        }
    ]  
    conversations.extend(behavior_list)
    # conversation = [{"from": "human", "value": "<video>\nObserve whether the child in the video exhibits the following behaviors:'cry', 'laugh', 'eat_hands', 'feed', 'eat', 'drink', 'lie', 'fall_backward', 'crawl', 'sit', 'on_feet', 'walk', 'dance', 'jump', 'run'. If any of these behaviors are present, please describe them with their corresponding start and end time points."},{"from": "gpt", "value": values[:-1]}]

    annotations.append({
        **base_data,
        "recipe_type": recipe_types,
        "segment": segments,
        "conversations":conversations
    })

    
    # # 处理未标注时间段
    # merged = merge_intervals(used_intervals)
    # all_times = sorted(time_slots.values())
    # total_range = [all_times[0], all_times[-1]]
    
    # gaps = []
    # prev_end = total_range[0]
    
    # ggap = 200
    # for interval in merged:
    #     if interval[0] > prev_end and ((interval[0]-prev_end) > ggap):
    #         #这里增加一个前后缩25帧的策略
    #         gaps.append([prev_end + int(ggap/8), interval[0]+ int(ggap/8)])
            
    #     prev_end = max(prev_end, interval[1])
    
    # if gaps:
    #     selected = random.choice(gaps)
    #     annotations.append({
    #         "id": f"{video_id}_other",
    #         **base_data,
    #         "recipe_type": "other",
    #         "segment": selected,
    #         "sentence": "other"
    #     })
    
    # return '\n'.join(json.dumps(ann) for ann in annotations)
    return annotations



if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
    # video_path = "/home/tc_workspace/data/children_actions/videos/20230711/1/videos_20230630_100725.mp4"
    # video_path = "/mnt/tos-tommi-algorithm/tc_workspace/data/videos/20230731/41/videos_20230720_2591.mp4"
    vllm_classs = VLLM()
    # vllm_classs=""
    
    root_dir = "/home/tc_workspace/code/VideoLLaMA3/data/annotations/"
    
    eaf_lists = find_eaf_files(root_dir)
    
    results = []
    n=0
    for eaf_file in tqdm(eaf_lists):
        try:
            result = get_total(eaf_file, vllm_classs)
            if result:
                results.extend(result)
                if n%10==0:    
                    save_to_jsonl(results, '/home/tc_workspace/code/VideoLLaMA3/data/child_llama3.jsonl')  
                    results = []
                n+=1  
        except Exception as e:
            print("------e", e)
            continue        

    # 生成JSONL文件
    
    
    # print(result)
