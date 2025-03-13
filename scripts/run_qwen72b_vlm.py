from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "/data0/tc_workspace/internlm/model/Qwen2.5-VL-72B-Instruct-AWQ"  # 替换实际模型路径

class VLLM:
    def __init__(self):
        # 初始化模型与分词器
        model_path = "/data0/tc_workspace/internlm/model/Qwen2.5-VL-72B-Instruct-AWQ"  # 替换实际模型路径
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 配置LLM引擎参数（网页1/3）
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",  # 支持混合精度加速（网页1）
            tensor_parallel_size=4,  # 多GPU并行（需≥2块GPU）
            gpu_memory_utilization=0.65,  # GPU内存利用率（网页3）
            max_model_len=19350,
            # enable_chunked_prefill=True,
            speculative_model_quantization="awq",
            swap_space=16  # CPU交换空间（网页2）
        )

        # 配置解码策略（网页1/7）
        self.sampling_params = SamplingParams(
            temperature=0.7,        # 控制随机性（0=确定性，1=随机）
            top_p=0.9,             # 核心词采样比例
            max_tokens=2048,       # 最大生成token数
            repetition_penalty=1.1 # 抑制重复生成（网页1）
        )
        
    def input_make(self, video_path, question_template):
        video_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": question_template},
                    {
                        "type": "video", 
                        "video": video_path,
                        "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
                    }
                ]
            },
        ]

        messages = video_messages
        
        processor = AutoProcessor.from_pretrained(model_path)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,

            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }
        
        return llm_inputs   
        
    def infer(self, video_path, question_template):
        
        llm_inputs = self.input_make(video_path)

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text

        return generated_text


if __name__ == "__main__":
    
    video_path = "/home/ubuntu/PublicData/children_actions/videos/20230823/175/videos_20230807_19809.mp4"
    question_template = "please describe the video content!"
    vllm_classs = VLLM()
    vllm_classs.infer(video_path=video_path, question_template=question_template)