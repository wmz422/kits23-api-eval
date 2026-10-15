# KiTS23 reasoning 与诊断 label 测试

在一台没有项目数据、没有训练环境的服务器上，下载固定版本 KiTS23 数据，调用你已经部署好的 **OpenAI 兼容视觉接口**，保存 reasoning/最终回答并计算诊断 label 准确率。默认 **200 张二维切片**，四类各 50 张，来自 160 个病例键；这里的 case 是一次切片请求，不是 200 位患者。

沿用原来的 `test_api.py`，提示词、解析和评分代码原样保留。模型部署由你负责；客户端无需 PyTorch、CUDA、OpenAI SDK 或 SAMTok 权重。

## 1. 准备客户端环境

需要 Linux、Bash、Python 3.10+（含 `venv`/pip）及网络。克隆本仓库后，在仓库根目录执行：

```bash
bash setup.sh
```

脚本在仓库内建立 `.venv`，只安装数据下载依赖 `huggingface_hub==0.36.0`。不会修改模型服务的环境。若服务器没有 Python，先用系统包管理器安装 Python 和 venv；Ubuntu/Debian 对应 `python3 python3-venv python3-pip`。也可指定已有 Python：`KITS23_PYTHON=/path/to/python bash setup.sh`。

## 2. 获得数据访问权限

数据来自 [microsoft/BiomedParseData](https://huggingface.co/datasets/microsoft/BiomedParseData)，当前需要登录该页面接受条件/申请访问，然后在服务器上配置该账号的只读 token。推荐交互登录，token 不进入仓库配置或 shell 历史：

```bash
.venv/bin/hf auth login
```

自动化运行可通过外部环境提供 `HF_TOKEN`。它用于下载数据；`OPENAI_API_KEY` 用于模型接口，两者用途不同。脚本不会代你接受数据使用条件。数据许可与下载权限以官方页面为准，当前标为 CC-BY-NC-SA-4.0；仓库不附带图像或 mask。

## 3. 下载数据并直接测试 200 条

模型由你指定，支持提供图像输入的 OpenAI 兼容 Chat Completions 接口。配置中的 `model` 默认为空；必须通过 `--model`、`API_MODEL` 环境变量或配置文件的 `api.model` 填写实际模型，优先级依次递减。未指定时会在下载数据或请求接口前报错。

将下面的地址和模型名改为你实际接口的值。`--model` 必须与服务提供的 model ID/别名一致；如需查看，可访问该服务的 `/v1/models`。

```bash
.venv/bin/python run.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model your-model-id
```

这条命令依次执行：检查配置 → 下载 `kits23.zip` → 校验归档 → 提取固定 200 条样本 → 校验每个文件 → 调用原 `scripts/test_api.py` → 输出总分和逐类指标。每条样本一次请求，无自动重试；默认并发 1、seed 42、最多输出 8192 tokens、每请求超时 300 秒。若接口有鉴权，从环境提供 `OPENAI_API_KEY`；无鉴权服务无需设置。

**首次下载约 16.18 GB**，不是只下载 200 张图片。官方提供的是完整 ZIP；脚本只解出本次需要的 **594 个文件，约 110.27 MB**（200 张原图及其病灶/肾脏参考 mask），不会下载其他 BiomedParse 子集。建议为 ZIP、提取文件和请求/响应预留至少 20 GB 空间。下载中断后重复同一命令可复用 Hugging Face 的部分下载；已准备好的文件再次逐一校验，通过后跳过网络及 ZIP 校验。

测试结果会保存在新建的 `results/run_<时间>/`，重复运行会保留旧批次。结尾会显示：

```text
Label accuracy: <正确数>/200 = <准确率>%
```

更换测试数量和并发：

```bash
.venv/bin/python run.py --base-url http://127.0.0.1:8000/v1 --model YOUR_MODEL_ID \
  --num-cases 20 --workers 2
```

数量范围 1–200；同一个 seed 按四类轮转均衡抽样。默认清单固定，不会在另一台机器重新随机划分患者或更换样本。也可以通过 `API_BASE_URL`、`API_MODEL` 环境变量覆盖示例配置，CLI 参数优先。

## 分步运行、离线数据和预检查

仅下载/准备数据：

```bash
.venv/bin/python scripts/prepare_data.py
```

已有完整官方 ZIP 时，直接使用，无需 Hugging Face 登录或网络：

```bash
python3 scripts/prepare_data.py --archive /path/to/kits23.zip
```

准备后的评分只需 Python 标准库。也可以把已准备好的 `data/extracted/` 复制到另一台机器，再执行上述准备命令的 `--verify-only` 模式重建当前机器清单。

预检查完整 200 条输入、保存选样计划，**不调用模型接口**：

```bash
python3 run.py --skip-download --dry-run --model your-model-id
```

`--dry-run` 本身默认也会准备数据；加 `--skip-download` 可强制禁用下载，缺数据则明确报错。计划在 `results/plan_<时间>/`。它只验证输入和配置，不验证模型服务是否已就绪。

在已准备好的数据上测试，强制不再下载：

```bash
python3 run.py --skip-download --base-url http://127.0.0.1:8000/v1 --model YOUR_MODEL_ID
```

`--data-dir /path/to/data` 可指定数据位置；`--output-dir /path/to/results` 可指定结果位置。CLI 路径相对当前目录，配置中的 `output_dir` 相对配置文件；脚本和清单默认路径相对仓库，因此从其他目录启动也可以。

## 配置文件

可复制 `configs/api.example.json` 为 `configs/api.local.json`，修改后执行：

```bash
python3 run.py --config configs/api.local.json
```

`api` 内设置 `base_url`、`model`、`api_key_env`、`timeout_seconds` 和 `request`；`test` 内设置 `num_cases`、`workers` 和 `seed`。密钥放在 `api_key_env` 指定的环境变量中，不写进 JSON。`api.request` 会原样传给 Chat Completions 接口，后端的 `chat_template_kwargs` 等也可在这里配置。

使用 vLLM 部署的 Qwen 视觉模型时，`base_url` 填服务的 `http://服务器:端口/v1`，`model` 填服务实际提供的模型 ID/别名。脚本的 `image_url` + base64 图片请求符合 [vLLM 多模态 Chat Completions 格式](https://github.com/vllm-project/vllm/blob/main/docs/features/multimodal_inputs.md)。服务若设置了 API key，通过 `OPENAI_API_KEY` 提供；无鉴权服务可留空。需要模型本身支持图像输入。

若后端要求 `max_completion_tokens`，用它替换 `max_tokens`，不要同时发送两个。Thinking 后端需要将最终 JSON 放入 `choices[0].message.content`；`reasoning_content` 会完整保存，但不会拿它猜 label。支持闭合的前置 `<think>...</think>` 和外层 JSON 代码围栏。截断或非 `stop` 结束会计为解析失败，可结合原始响应调整 token/超时配置后新建批次。

## 输出及评分

模型每次只收到固定中性问题和原始 PNG；不收到 GT、mask、病例路径或类别提示。要求最终返回：

```json
{
  "findings": "可见观察",
  "assessment": "与证据关联的简短解释",
  "label": "tumor_only"
}
```

合法 label：`tumor_only`（仅肿瘤）、`cyst_only`（仅囊肿）、`both`（两类共存）、`neither`（本切片两类均未标注）。

| 产物 | 内容 |
|---|---|
| `summary.json` | 正确数/全部样本数、逐类准确率、混淆矩阵、错误数、仅有效预测准确率 |
| `results.jsonl` | 每条 GT/预测 label、是否正确、状态、耗时、usage |
| `cases/00001/request.json` | 实际请求，含原图 base64，不含鉴权头 |
| `cases/00001/response.txt` | 完整响应原始字节，包括 reasoning_content、最终 content 或 HTTP 错误体 |
| `cases/00001/result.json` | 解析后的回答和该条评分/失败原因 |
| `selected_cases.jsonl` | 本批实际样本、图像和参考 mask 路径 |
| `config.redacted.json`、`metadata.json` | 实际配置、种子、数据版本、代码/清单哈希、启动命令与环境 |
| `run.log`、`exit.json` | 进度和退出状态 |

网络未返回任何响应时没有 `response.txt`，原因仍保留在 `result.json`。主要指标是**四状态严格匹配正确数/全部选中样本数**，网络失败、非法 JSON/label、截断均计错；不先筛掉错误再计准确率。只自动评分最终 label，不自动认定 reasoning 的医学正确性；本次不接第二次 GT 审核，不输出预测分割。

正常完成且至少有一个有效 label 时退出 0；全部无效退出 1；配置/准备错误退出 2。部分失败可能退出 0，应同时查看 `summary.json` 的 `status_counts`。

这批数据是既有 **train 候选清单**，不是独立 heldout benchmark。`neither` 只依据现有二维导出标注，原始三维 GT 完整性仍待核验；分数表示与导出标签的一致率，不代表患者层面的临床诊断准确率。

## 4. 可视化已保存的结果

安装查看器的可选依赖（仅增加 Pillow；不需要 GPU，不会调用模型）：

```bash
bash setup.sh --viewer
.venv/bin/python scripts/results_viewer.py --port 8507
```

打开 **http://127.0.0.1:8507**。默认读取仓库 `results/` 下的运行批次，支持：

- 选择批次、搜索样本、按 GT 类别或正确/错误/失败筛选，以及前后/键盘导航。
- 原图和逐类别 **GT mask 叠加**，调整透明度、切换肿瘤/囊肿和可选肾脏参考。
- 模型最终 JSON、GT/预测 label、是否正确、耗时和 usage。
- 完整原始响应默认折叠，展开后才加载，保留 `reasoning_content`；切换样本/批次重新折叠。
- 运行过程中查看已返回样本；新批次生成后点击“刷新”即可发现。

页面中的 mask 来自已有标注，当前模型 API 没有输出预测 mask。不会修改或重新评分已保存的结果。

自定义结果根目录和数据目录：

```bash
.venv/bin/python scripts/results_viewer.py \
  --root /path/to/results --data-dir /path/to/data --port 8507
```

`--root` 是包含各个 `run_*` 目录的父目录。`--data-dir` 是数据准备目录，内部应有 `extracted/kits23/train/` 和 `train_mask/`。若保存的绝对图像路径在新机器不存在，会从该数据目录按原文件名查找，再核验原 SHA-256；不需要改写历史结果。CLI 的显式相对路径以启动目录为准，默认目录以仓库为准。

需要后台运行时使用启动器，它会打印 PID、日志和 URL，并检查服务就绪：

```bash
.venv/bin/python scripts/start_results_viewer.py --port 8507
```

后台记录默认在 `results/viewer_sessions/`；可用 `--name` 指定新名称，`--output-root` 指定日志位置。前台运行按 Ctrl+C 停止；后台运行可用启动记录中的 PID 停止该进程。已有端口被占用时换 `--port`，启动器不会停止旧服务或覆盖旧记录。

查看器默认只监听 `127.0.0.1`。在服务器运行、从自己电脑浏览时，可在自己电脑上建立 SSH 转发：

```bash
ssh -L 8507:127.0.0.1:8507 your-user@your-server
```

然后在电脑打开 http://127.0.0.1:8507。需要直接监听服务器网卡时可显式增加 `--host 0.0.0.0`。

## 数据版本和本地验证

`manifests/source.json` 固定官方 revision、ZIP 字节数和 SHA-256；`manifests/cases_200.jsonl` 固定每张图像/mask 的 SHA-256。`PROVENANCE.json` 记录原脚本和原清单的来源校验值。`data/prepared.json` 保存本机实际数据准备记录。原脚本没有改动，也没有复制旧的接口配置或密钥。

查看器源自既有结果页面，源码来源与校验值同样记录在 `PROVENANCE.json`。查看器测试在安装 `--viewer` 依赖后运行；未安装 Pillow 时跳过这些可选测试。

运行无需数据、无需联网的工程测试（合成 PNG 和本地模拟 HTTP；不调用真实模型）：

```bash
python3 -m unittest discover -s tests -v
```

## 放到自己的 Git 服务器

该目录本身就是可独立使用的仓库。已有 Git 历史时，设置你自己的远端后推送：

```bash
git remote add origin YOUR_GIT_URL
git push -u origin main
```

另一台服务器随后 `git clone YOUR_GIT_URL` 并从第 1 步开始。未配置远端时也可以直接解压交付包使用。`.gitignore` 已排除数据、结果、环境、`.env` 和本地配置；默认不会把大文件和凭据提交到 Git。
