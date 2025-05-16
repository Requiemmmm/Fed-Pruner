**复现目标:** 复现论文中 Table I, II, III, IV 展示的关于 Fed-Pruner 在 GLUE (SST-2, QNLI, QQP, MNLI) 和 SQuAD 数据集上，基于 BERT-base 模型进行联邦剪枝的实验结果。

**核心流程概述 (根据 readme.txt):** 

1. **配置与修改:** 根据目标数据集（GLUE 的四个子集或 SQuAD）修改 `pipeline` 文件夹下的 `entry.py` 和 `fed_entry.py` (或 `qa_entry.py` 和 `fed_qa_entry.py`) 文件。
2. **训练教师模型:** 运行 `glue_teachermodel_train.py` (GLUE) 或 `squad_teachermodel_train.py` (SQuAD)，观察 5 个 epoch 的结果，保存效果最好的 checkpoint 并重命名（例如 `sst2-half-datas`）。
3. **运行联邦剪枝:** 修改 `args.py` (GLUE) 或 `qa_args.py` (SQuAD) 中的参数（主要是 `half` 和 `distill`），然后运行 `glue_fedmodel_train.py` (GLUE) 或 `squad_fedmodel_train.py` (SQuAD)。

**详细复现步骤:**

**〇、准备工作**

1. 环境配置:
   - 必要的库: PyTorch, Transformers, Datasets, NumPy(见requirements)

**一、训练教师模型 (对应 Table I 中的 'BERTbase w/ half data' 行)**

- **目标:** 训练一个在**halfdata**上 fine-tune 的 BERT-base 模型，作为后续知识蒸馏的教师模型。

- 步骤:

  1. 为特定数据集修改代码:

     - GLUE 数据集 (SST-2, QNLI, QQP, MNLI):

       - 打开 `pipeline/entry.py`。

       - **修改数据集加载路径:** 找到 `load_from_disk('./datasets/mnli')` 这一行，将其中的 `'./datasets/mnli'` 更改为目标数据集的路径，例如 `'./datasets/sst2'`。

       - 修改 Tokenize 函数:

          根据 

         ```
         entry.py
         ```

          文件中的注释，修改 

         ```
         tokenize_function
         ```

          内部的 

         ```
         tokenizer()
         ```

          调用，使其匹配目标数据集的列名。

         - SST-2: `tokenizer(example["sentence"], truncation=True)`
         - QQP: `tokenizer(example["question1"], example["question2"], truncation=True)`
         - QNLI: `tokenizer(example["question"], example["sentence"], truncation=True)`
         - MNLI: `tokenizer(example["premise"], example["hypothesis"], truncation=True)` (默认)

       - **修改验证集过滤:** 根据注释，调整 `dataset['validation'] = ...` 行。MNLI 使用 `validation_matched`，其他三个使用 `validation`。

       - **确认使用一半数据:** 确保 `dataset['train'] = dataset['train'].shard(num_shards=2, index=0, contiguous=True)` 这一行存在且未被注释，它指定了只使用一半训练数据。

     - SQuAD 数据集:

       - 打开 `pipeline/qa_entry.py`。
       - **修改数据集加载路径:** 找到 `load_from_disk('./datasets/squad')`，确认路径正确。
       - **确认使用一半数据:** 找到 `datasets['train'] = datasets['train'].shard(num_shards=2, index=0, contiguous=True)` 这一行，确保其存在且未被注释。

  2. 运行训练脚本:

     - **GLUE:** 在终端中运行 `python glue_teachermodel_train.py`。
     - **SQuAD:** 在终端中运行 `python squad_teachermodel_train.py`。

  3. 保存最佳模型:

     - 训练过程会持续 5 个 epoch，并每个 epoch 保存一个 checkpoint 在 `[glue]` (或 `[squad]`，具体看 `args.py`/`qa_args.py` 中的 `output_dir`) 文件夹下的 `checkpoint-xxxx` 子目录中。
     - 观察每个 epoch 输出的评估结果 (Accuracy 或 F1)。
     - 找到效果最好的 epoch 对应的 checkpoint 文件夹。
     - 将该文件夹重命名为 `[数据集名]-half-datas`，例如 `sst2-half-datas` 或 `squad-half-datas`。将其移动到 `[glue]` 或 `[squad]` 目录下（根据 `fed_entry.py` 或 `fed_qa_entry.py` 中的加载路径决定，代码中似乎固定在 `[glue]` 目录下加载）。
     - (可选) 删除其他不必要的 checkpoint 文件夹以节省空间。

**二、运行联邦剪枝 (Fed-Pruner)**

- **目标:** 使用联邦学习框架对模型进行剪枝，复现 Table I, II, III, IV 的结果。
- **通用步骤 (每次运行前):**
  1. 为特定数据集修改代码:
     - GLUE 数据集 (SST-2, QNLI, QQP, MNLI):
       - 打开 `pipeline/fed_entry.py`。
       - **修改数据集加载路径:** 找到所有 `load_from_disk('./datasets/mnli')` 的地方，修改为目标数据集路径 (例如 `./datasets/sst2`)。
       - **修改 Tokenize 函数:** 找到 `tokenize_function`，根据目标数据集修改 `tokenizer()` 调用（同步骤 `一、1`）。
       - **修改验证集过滤:** 找到 `dataset['validation'] = ...` 行，根据目标数据集进行修改（同步骤 `一、1`）。
       - **修改模型加载路径:** 找到加载 `t_model` 和 `s_model` 的行 (例如 `TModel.from_pretrained('./[glue]/mnli-half-datas')`)，将路径中的数据集名修改为当前目标数据集对应的教师模型名称 (例如 `./[glue]/sst2-half-datas`)。
     - SQuAD 数据集:
       - 打开 `pipeline/fed_qa_entry.py`。
       - **修改数据集加载路径:** 确认 `load_from_disk('./datasets/squad')` 路径正确。
       - **修改模型加载路径:** 找到加载 `t_model` 和 `s_model` 的行 (例如 `TModel.from_pretrained('./[glue]/squad-half-datas')`)，确认路径正确。
  2. 运行联邦剪枝脚本:
     - **GLUE:** 在终端中运行 `python glue_fedmodel_train.py`。
     - **SQuAD:** 在终端中运行 `python squad_fedmodel_train.py`。
  3. **观察结果:** 训练过程会持续多个 epoch (代码中 `Server` 类默认为 100)。每个 epoch 结束后会进行评估并打印结果，包括精度 (accuracy/f1) 和稀疏度 (sparsity)。注意 readme 中提到“每过一个epoch会print符合压缩条件(参数量为原参数量的10%)的最好的结果”。论文中的结果通常是收敛后的最佳结果 (在稀疏度约为 10% 时)。
- **具体实验配置 (修改参数并运行):**
  - **复现 Table I - Fed-Pruner w/ distillation:**
    - 修改参数文件:
      - GLUE: `pipeline/args.py` -> 设置 `distill = True`, `half = True` (根据readme，`half` 控制联邦学习过程是否只用客户端数据的后一半)。
      - SQuAD: `pipeline/qa_args.py` -> 设置 `distill = True`, `half = True`。
    - **运行:** 执行上述 **通用步骤 1 和 2**。记录最终稳定在 10% 稀疏度左右的最佳性能。
  - **复现 Table I - Fed-Pruner w/o distillation:**
    - 修改参数文件:
      - GLUE: `pipeline/args.py` -> 设置 `distill = False`, `half = True`。
      - SQuAD: `pipeline/qa_args.py` -> 设置 `distill = False`, `half = True`。
    - **运行:** 执行上述 **通用步骤 1 和 2**。记录最终结果。
    - **注意:** 此时加载 `s_model` 和 `t_model` 的路径需要修改。根据 `fed_entry.py`/`fed_qa_entry.py` 中的 `if self.distill == False:` 逻辑，此时应加载原始的预训练模型，即路径应为 `'./model'`。
  - **复现 Table II - Fed-Pruner w/o distillation w/ all data:**
    - 修改参数文件:
      - GLUE: `pipeline/args.py` -> 设置 `distill = False`, `half = False`。
      - SQuAD: `pipeline/qa_args.py` -> 设置 `distill = False`, `half = False`。
    - **运行:** 执行上述 **通用步骤 1 和 2**。确保模型加载路径为 `'./model'`。记录最终结果。
  - **复现 Table III - 5 Clients:**
    - 修改参数文件:
      - GLUE: `pipeline/args.py` -> 设置 `distill = True`, `half = True`。
      - SQuAD: `pipeline/qa_args.py` -> 设置 `distill = True`, `half = True`。
    - 修改核心逻辑文件:
      - GLUE: `pipeline/fed_entry.py` -> 在 `Client` 和 `Server` 类的 `__init__` 方法中，找到 `num_clients = 2`，将其修改为 `num_clients = 5`。
      - SQuAD: `pipeline/fed_qa_entry.py` -> 同上修改 `num_clients`。
    - **运行:** 执行上述 **通用步骤 1 和 2**。确保模型加载路径为对应的教师模型。记录最终结果。
    - **注意:** 修改完后记得改回 `num_clients = 2` 以进行其他实验。
  - **复现 Table IV - Fast Sparsity Variation Rate:**
    - 修改参数文件:
      - GLUE: `pipeline/args.py` -> 设置 `distill = True`, `half = True`。
      - SQuAD: `pipeline/qa_args.py` -> 设置 `distill = True`, `half = True`。
    - 修改核心逻辑文件:
      - GLUE: `pipeline/fed_entry.py` -> 在 `Server` 类的 `run` 方法中，找到更新 `target_sparsity` 的行：`self.client.distill_args.target_sparsity = max(0.1, self.client.distill_args.target_sparsity - 0.2)`，将 `- 0.2` 修改为 `- 0.4`。
      - SQuAD: `pipeline/fed_qa_entry.py` -> 在 `Server` 类的 `run` 方法中，找到更新 `target_sparsity` 的行：`self.client.distill_args.target_sparsity = max(0.2, self.client.distill_args.target_sparsity - 0.2)`，将 `- 0.2` 修改为 `- 0.4`。（注意SQuAD的最低稀疏度是0.2）。
    - **运行:** 执行上述 **通用步骤 1 和 2**。确保模型加载路径为对应的教师模型。记录最终结果。
    - **注意:** 修改完后记得改回 `- 0.2` 以进行其他实验。

**三、结果验证**

- 将运行得到的结果 (Accuracy/F1 Score 和最终的 Sparsity) 与论文中对应表格的数值进行比较。

**重要提示:**

- **数据集修改:** 复现不同数据集结果时，**务必**仔细检查并修改 `pipeline/entry.py` 和 `pipeline/fed_entry.py` (或 `qa_entry.py` / `fed_qa_entry.py`) 中所有与数据集相关的路径和处理逻辑（如 `load_from_disk`, `tokenize_function`, 验证集选择等）。这是最容易出错的地方。
- **教师模型:** 对于需要蒸馏的实验，确保教师模型已正确训练并放置在 `fed_entry.py` 或 `fed_qa_entry.py` 指定的加载路径下。
- **参数配置:** 仔细检查 `args.py` 或 `qa_args.py` 中的 `distill` 和 `half` 参数是否符合当前实验场景。
- **代码修改:** 复现 Table III 和 IV 需要直接修改 `.py` 文件中的核心逻辑，完成后记得恢复原状。
- **计算资源:** 训练和剪枝过程计算量大，需要较长时间和 GPU 资源。
- **论文细节:** 仔细阅读论文 III.B 节和 IV.A/B/C/D 节，了解 Fed-Pruner 的具体实现细节（如 Loss 函数构成、稀疏度计算、差分隐私机制等）