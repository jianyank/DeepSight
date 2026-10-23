# 原始基线工程补齐记录

核查日期：2026-10-23。范围：原始方法的依赖、入口和数据注册；不实施优化方案，不运行训练、推理、测试或安装。

## 结论与源码边界

本轮按作者 GitHub 仓库 hotdogcheesewhite/DeepSight 的 main tree 核对，本地 HEAD 与返回的 tree SHA 均为 2ad3cb7964949ea1a8f735514721e27a79563c0a；公开 tracked 文件没有发现本地漏下载。问题是公开工程仍依赖作者环境及外部资源，不是缺少一整套可直接复制的程序。

当前仅补齐已识别的工程衔接缺口，**不等于完整复现已完成，也不保证现在可启动**。原方法的模型、损失、数据处理算法和训练超参数保持不变。

## 已补内容

| 文件 | 处理 |
|---|---|
| requirements.txt | 补回 setup.py / MANIFEST.in 引用但原仓库缺失的清单；依据 example.txt 与源码 imports 重建，不是作者锁文件，未进行依赖解析。 |
| scripts/train/train.sh | Linux/CUDA 主入口；定位仓库根目录，优先导入本地修改版 Transformers，只启动一次 torchrun，YAML 路径作为位置参数。默认 8 进程，可用 NPROC_PER_NODE 覆盖。 |
| scripts/train/train.bat | Windows cmd 参数转发入口，默认 1 进程；不代表 DeepSpeed / FlashAttention 在原生 Windows 上可用。 |
| nebula.sh | 保留兼容入口，转交 scripts/train/train.sh；移除原先连续启动两次训练及错误的 configs 路径、--config 参数。 |
| data/dataset_info.json | 增加 bench2drive_bev_target 的本地 JSONL 注册约定；不生成假数据。 |
| tests/test_baseline_layout.py | 新增注册项、入口与修改版 Transformers 版本的文件级回归检查；遵从用户要求，未执行。 |

证据：src/llamafactory/hparams/parser.py 的 read_args 直接读取首个 YAML/JSON 位置参数；src/transformers/src/transformers/__init__.py 标注版本 4.56.0.dev0；config/ad_bev_v3.yaml 和 ad_bev_v4.yaml 引用 bench2drive_bev_target。

## 环境准备顺序（仅说明，未执行）

原作者流程面向 Linux / NVIDIA CUDA。README 与 example.txt 的 PyTorch 版本信息不完全相同；后者记录 torch 2.7.1、torchvision 0.22.1、DeepSpeed 0.16.9、FlashAttention 2.8.1，不能据此认定为验证过的组合。先根据真实服务器准备匹配的 Python、CUDA、PyTorch / torchvision、DeepSpeed、FlashAttention。默认配置启用了 bf16、fa2、DeepSpeed，并使用 TensorBoard。未给出通吃硬件的安装命令，也未改变配置来绕过这些要求。

不要直接安装 PyPI Transformers 替换本地修改版，否则会丢失 DeepSight 模型实现。硬件环境就绪后，包安装顺序如下（Windows cmd 路径示例；不是安装 CUDA 栈或验证 Windows 支持的命令）：

~~~bat
cd /d D:\Python\Project\DeepSight
python -m pip install -e .\src\transformers
python -m pip install -e .
~~~

第二条会读取根 requirements.txt。训练、vLLM 推理、CARLA / Bench2Drive 依赖应分别核对，不能把 example.txt 中的混合环境快照当作训练锁文件。默认不在训练环境安装 vLLM。原闭环文档采用 CARLA 0.9.16；这里只记录仓库方案，没有验证适配。

## 仍必须准备的真实资源

1. **数据**：data/bench2drive_bev_target.jsonl 当前不存在。新增注册键只约定这个本地路径；可改为真实 JSONL 路径。每行含 messages（role/content 对话）与 images（图像路径列表），采用 sharegpt 映射，角色为 user / assistant。须沿用原生成流程的 BEV token、轨迹、CoT 格式，不可用普通对话样例替代训练数据。
2. **五帧监督**：src/llamafactory/data/ad_collator.py 的 ADCollector 从 images 尾部取出五张未来 BEV 图像，按 256×256 归一化；标签需含匹配的 start/end BEV token 和相应长度。注册成功并不证明数据满足此约束。
3. **模型与教师**：config/ad_bev_v4.yaml 的 model_name_or_path 仍指向作者 NAS checkpoint。模型配置必须提供有效的 dinov3_config 文件，checkpoint 必须包含正确的 DINOv3 教师权重及匹配的模型结构 / tokenizer / 特殊 token。模型构造中的 _from_config 与 requires_grad_(False) 只构建并冻结教师，**不能代替加载预训练教师权重**。原 tests/qwen_2.5_vl_with_dino.py 中相关加载代码含注释实验分支，不能当作从普通 Qwen 自动生成完整 DeepSight 权重的可用入口。
4. **路径与缓存**：检查模型路径、tokenized_path、图片路径和 output_dir；默认 tokenized_path 仍是作者 NAS 缓存，不能沿用不明来源的旧缓存。保留原配置以免擅自改变实验，运行前由实际资源位置决定修改。
5. **闭环资源**：准备 CARLA、地图、Bench2Drive 数据及路线、推理模型和对应环境；CoT 标注来源、数据切分、权重版本与授权仍待确认。

## 下一步与验收边界

下一步先取得上述资源并固定环境、配置、数据版本；经用户同意后再做导入检查、单批次前向与小规模干净基线，然后才开始优化。记录缺失 / 意外初始化的权重，特别是 DINOv3，不能仅以训练命令未报错认定教师有效。

本轮只做文件回读、JSON 文本解析、路径核对和 Git diff 静态检查；未执行新增 pytest 文件，未验证依赖求解、导入、训练、推理或论文指标。原始源码其他硬编码路径可能仍存在，不将本记录视为全量运行审计。
