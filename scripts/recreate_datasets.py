from moviepy.editor import VideoFileClip
import cv2
import subprocess

def cut_video_moviepy(video_path, segment, output_path="output.mp4"):
    """基于帧数精准切割的高效方案（耗时约原视频时长1/8）"""
    
    fps = 1000
   
    # 计算时间区间（网页2/5）
    start_sec = segment[0] / fps
    end_sec = segment[1] / fps
    
    # 执行切割（网页1/5）
    with VideoFileClip(video_path, fps_source='fps') as video:
        # 自动校正超界时间
        end_sec = min(end_sec, video.duration)
        # 带异常处理的稳健写法
        try:
            clip = VideoFileClip(video_path).subclip(start_sec, end_sec)
            clip.write_videofile("output.mp4", audio_codec='aac')  # 必须指定音频编码[6](@ref)
      
        except AttributeError:
            # 新版API调用方式
            from moviepy.video.fx.all import crop
            clip = VideoFileClip("input.mp4")
            processed_clip = crop(clip, y1=0, y2=clip.h, x1=0, x2=clip.w)
            processed_clip = processed_clip.subclip(10, 20)# 带异常处理的稳健写法
            



# # get_precise_duration("/home/ubuntu/PublicData/children_actions/videos/20230823/175/videos_20230807_19809.mp4")
# # # 调用示例（网页5）
# cut_video_moviepy(
#     video_path="/home/ubuntu/PublicData/children_actions/videos/20230823/175/videos_20230807_19809.mp4",
#     segment=[2299, 5033],
#     output_path="crawl_segment.mp4"
# ) 


import json
import os
import time
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import rotate
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化日志系统
        self.log_file = os.path.join(output_dir, "process.log")
        open(self.log_file, 'w').close()  # 清空日志文件

    def _log(self, message):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def process(self, jsonl_path, retry=3):
        """处理入口"""
        with open(jsonl_path) as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            
            with tqdm(total=total_lines, desc="Processing") as pbar:
                for line in f:
                    data = json.loads(line.strip())
                    self._process_item(data, retry)
                    pbar.update(1)

    def _process_item(self, data, retry):
        """处理单个条目"""
        for attempt in range(retry):
            try:
                video_id = data["id"]
                src_path = data["video_path"]
                start, end = data["segment"]
                
                # 验证文件存在性
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"视频文件不存在: {src_path}")
                
                with VideoFileClip(src_path) as video:
                    # 自动容错处理
                    if video.duration == 0:
                        raise ValueError("视频时长为零")
                        
                    fps = 1000  # 默认帧率
                    start_time = max(0, start / fps)
                    end_time = min(end / fps, video.duration)
                    
                                        # 检测旋转元数据
                    if video.rotation in (0, 180):
                        # 物理旋转视频流
                        video = video.resize(video.size[::-1]).set_position((0,0))
                        video = rotate(video, -video.rotation)
                                    
                    # 执行切割
                    output_dir = src_path.replace("videos", "videos_cut")
                    output_dir = os.path.dirname(output_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    subclip = video.subclip(start_time, end_time)
                    output_path = os.path.join(output_dir, f"{video_id}.mp4")
                    
                    # 优化编码参数
                    subclip.write_videofile(
                        output_path,
                        codec="libx264",
                        preset="medium",
                        threads=64,
                        audio_codec="aac",
                        temp_audiofile="temp-audio.m4a",
                        remove_temp=True,
                        ffmpeg_params=['-movflags', '+faststart']
                        
                    )
                
                self._log(f"Success: {video_id}")
                return
                
            except Exception as e:
                if attempt == retry -1:
                    self._log(f"Failed after {retry} attempts: {video_id} - {str(e)}")
                else:
                    time.sleep(2 ** attempt)  # 指数退避重试
import json
import logging
import os
from multiprocessing import Pool
import ffmpeg
from run_qwen72b_vlm import VLLM

question_template = \
"""
Create a dataset for me, following this format.
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nDescribe this video."
      },
      {
        "from": "gpt",
        "value": "<answer1>"
      },
      {
        "from": "human",
        "value": "<question2>"
      },
      {
        "from": "gpt",
        "value": "<answer2>"
      },
      {
        "from": "human",
        "value": "<question3>"
      },
      {
        "from": "gpt",
        "value": "<answer3>"
      }
    ]
  }
]
The questions and answers, please generate for me, based on the video I sent to you. Thes questions should be from the shallow to the deep, and the answers should be as detailed and correct as possible. The questions and answers should be stick to the contents in the video itself, like objects, peoples, equipment, environment, purpose, color, attitude, etc. 3 question and answer pairs.
"""


class ffmpegg():
    def __init__(self):
        self.logger = self.setup_logger()
        # self.vllm_classs = VLLM()

    # 日志配置模块
    def setup_logger(self):
        """配置多级别日志输出"""
        logger = logging.getLogger('VideoProcessor')
        logger.setLevel(logging.DEBUG)
        
        # 文件处理器（按日切割）
        file_handler = logging.FileHandler('video_processing.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def process(self, jsonl_path, retry=3):
        """处理入口"""
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line.strip())
                video_id = data["id"]
                src_path = data["video_path"]
                start, end = data["segment"]
                
                # 验证文件存在性
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"视频文件不存在: {src_path}")
                
    def convert_time(self,ms):
        """将毫秒转换为FFmpeg时间格式[5](@ref)"""
        seconds = ms / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"            

    def read_jsonl(self, file_path):
        """带异常处理的JSONL文件读取[1,3](@ref)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line.strip())
        except Exception as e:
            self.logger.error(f"JSONL文件读取失败: {str(e)}")
            return []

    def process_video(self, item):
        """GPU加速视频处理核心"""
        try:
            input_path = item["video_path"]
            output_name = f"{item['id']}.mp4"
            start, end = item["segment"]
            
            # 时间格式转换
            ss = self.convert_time(start)
            to = self.convert_time(end)
            
            # 构建GPU加速命令[3,6](@ref)
                                # 执行切割
            output_dir = input_path.replace("videos", "videos_cut")
            output_dir = os.path.dirname(output_dir)
            output_path = os.path.join(output_dir, output_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(output_path):
                stream = ffmpeg.input(input_path, ss=ss, to=to, hwaccel='cuda')
                stream = ffmpeg.output(
                    stream,
                    output_path,
                    vcodec='h264_nvenc',
                    **{'b:v': '8M', 'bufsize': '16M'},
                    acodec='aac',
                    loglevel='warning'
                )
                
                ffmpeg.run(stream, overwrite_output=True)
                self.logger.info(f"成功处理: {input_path} -> {output_path}")
            return True
            
        except ffmpeg.Error as e:
            self.logger.error(f"处理失败[{input_path}]")
            return False
        except KeyError as e:
            self.logger.error(f"数据字段缺失: {str(e)}")
            return False

    def batch_processing(self, jsonl_path, workers=4):
        """多线程批处理调度"""
        items = list(self.read_jsonl(jsonl_path))
        if not items:
            self.logger.critical("无有效处理条目")
            return
        
        with Pool(processes=workers) as pool:
            results = pool.map(self.process_video, items)
            success_rate = sum(results)/len(results)
            self.logger.info(f"处理完成 成功率: {success_rate:.1%}")
            
    def recaption(self):
        items = list(self.read_jsonl(jsonl_path))
        try:
            for item in items:
                input_path = item["video_path"]
                output_name = f"{item['id']}.mp4"
                
                output_dir = input_path.replace("videos", "videos_cut")
                output_dir = os.path.dirname(output_dir)
                output_path = os.path.join(output_dir, output_name)
                
                if os.path.exists(output_path):
                    question_template = question_template
                    generated_text = self.vllm_classs.infer(video_path=output_path, question_template=question_template)
        except Exception as e:
            pass    


if __name__ == "__main__":
    # # 使用示例
    data_root = "/data0/tc_workspace/internlm/code/VideoLLaMA3/data"
    ff = ffmpegg()
    for case in tqdm(["train", "val", "test"]):
        jsonl_path = data_root+"/child_llama3_%s.jsonl"%(case) 
        # 主流程--生成裁剪后的视频
        ff.batch_processing(jsonl_path)
        #生成caption
        