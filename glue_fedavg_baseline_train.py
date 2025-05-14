'''’from pipeline.fed_entry import Server
def main():
    Server().run()
if __name__== "__main__" :
    main()'''

# 这种方式不太常见，因为参数通常在调用入口处解析,为了保持原来的结构，假设 fed_baseline_entry.py内部处理参数 ---

'''from pipeline.fed_baseline_entry import Server # 假设 Server 类内部会解析参数

def main():
    Server().run() # 这种方式下，Server 类需要自己处理来自 baseline_args.py 的参数解析

if __name__ == "__main__":
    main()'''

# 导入我们创建的 baseline 版本的联邦入口逻辑
# 假设 fed_baseline_entry.py 文件中有一个 main 函数来处理启动流程
from pipeline import fed_baseline_entry 

# 导入 HfArgumentParser 用于解析我们定义的 dataclass 参数
from transformers import HfArgumentParser 
# 导入我们定义的 baseline 参数类
from pipeline.baseline_args import ModelArguments, TrainingArguments # 确保是从 baseline_args.py 导入

def main():
    # 1. 解析参数
    # 使用 HfArgumentParser 来解析 ModelArguments 和 TrainingArguments
    parser = HfArgumentParser((ModelArguments, TrainingArguments)) 
    # 确保 model_args 和 training_args 都被接收
    model_args, training_args = parser.parse_args_into_dataclasses() 

    print("--- Parsed Baseline Model Arguments ---")
    print(model_args) 
    print("--- Parsed Baseline Training Arguments ---")
    print(training_args) 
    print("-----------------------------------------")
    
    # 2. 调用 baseline 版本的联邦训练主函数，并传入 *两个* 参数对象
    # *** 修改处：传递两个参数 ***
    fed_baseline_entry.main(model_args, training_args) 

if __name__ == "__main__":
    main()