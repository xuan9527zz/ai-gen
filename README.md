# Illustrious Reconstruction Studio

本地运行的动漫图像 **分析 → Prompt 重建 → ComfyUI 生图 → 人工更正/评分** 工作台。

当前打包版本：**v0.5.3**

> 设计目标：模型和个人数据留在自己的电脑上；电脑负责 GPU 推理，手机只作为控制界面。

## 已有功能

- VLM + WD14 并行图像分析
- VLM 风格、渲染、材质与光照词的确定性漏项恢复
- GIF / APNG / animated WebP 在进入 WD14 前自动取首帧
- Danbooru Source Tags 清洗
- Pixiv 日文 Tags 映射
- Pixiv unknown tag → 本地 LLM 语义映射 → NAIDv3 Tag Search exact-match / Power 验证
- Analyzer Final Prompt 与 provenance 保存
- SQLite 保存多次 Analysis Run
- ComfyUI API 自动生成
- BASE / REFINER 两段式 workflow
- 3 个可开关 LoRA 槽位
- 1 / 5 / 10 个不同 seed 候选图
- 固定 Prompt / seed 的 A–E 生成质量诊断
- 自然语言 Prompt 更正
- Manual Positive Prompt
- 原图 / 生成图对比
- 1–5 分评分与备注
- 手机局域网 UI

## 项目结构

```text
illustrious-reconstruction-studio/
├─ app/
│  ├─ analyzer.py
│  ├─ generator.py
│  ├─ database.py
│  ├─ image_inputs.py
│  ├─ source_tags.py
│  ├─ naid_verifier.py
│  └─ web_ui.py
├─ config/
│  ├─ generation.example.json
│  ├─ generation.json              # 本地配置，不提交 Git
│  └─ jp_to_danbooru_tags.json
├─ workflows/
│  ├─ anime.example.json
│  ├─ anime.json                   # 你的 ComfyUI API workflow，不提交 Git
│  └─ wd14_api.json
├─ scripts/
│  ├─ setup_local.py
│  ├─ check_env.py
│  ├─ start_studio.bat
│  └─ start_studio.ps1
├─ tests/
│  ├─ test_generation_diagnostics.py
│  └─ test_image_inputs.py
├─ data/                            # 数据库/原图/生成图/缓存，不提交 Git
├─ .gitignore
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## 1. 环境要求

你需要自行准备并运行：

- Windows 10/11
- Python 3.10+
- Ollama：默认 `http://127.0.0.1:11434`
- ComfyUI：默认 `http://127.0.0.1:8188`
- WD14 Tagger ComfyUI 节点
- VLM / merger 模型

当前 Analyzer 默认模型：

```text
VLM:
hf.co/mradermacher/Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF:Q5_K_M

Merger:
qwen3:8b
```

模型名可以用环境变量覆盖：

```text
ILLUSTRIOUS_VLM_MODEL
ILLUSTRIOUS_MERGER_MODEL
ILLUSTRIOUS_PIXIV_MODEL
ILLUSTRIOUS_CORRECTION_MODEL
ILLUSTRIOUS_OLLAMA_URL
ILLUSTRIOUS_COMFY_URL
```

## 2. 首次安装

在仓库目录打开 PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
py scripts\setup_local.py
```

然后把你自己已经验证好的 ComfyUI **API Format JSON** 保存为：

```text
workflows\anime.json
```

项目里的 `anime.example.json` 只是一个不含个人 LoRA 的基础模板。

如果你的节点 ID 与示例不同，修改：

```text
config\generation.json
```

里的 `workflow_nodes`。

## 3. 启动

先启动：

1. Ollama
2. ComfyUI

然后双击：

```text
scripts\start_studio.bat
```

或 PowerShell：

```powershell
.\scripts\start_studio.ps1
```

电脑浏览器：

```text
http://127.0.0.1:8765
```

## 4. 手机控制

电脑和手机在同一个 Wi-Fi / 局域网时，Studio 启动后会显示：

```text
Phone/LAN: http://192.168.x.x:8765
```

手机 Safari / Chrome 打开该地址即可。

### Windows 防火墙

如果首次访问被拦截，只允许 Python 在 **专用网络（Private network）** 通信即可。

### 不要直接暴露公网

Studio 当前没有账号登录系统，因此：

**不要把 8765 端口直接做路由器公网端口映射。**

以后若需要在外面用 4G/5G 控制家里的电脑，推荐加 **Tailscale**，把手机与电脑放在私有网络中。

## 5. Pixiv Source Tags

网页中：

```text
Source type = pixiv
```

然后原样粘贴 Pixiv 日文 Tags。

已知映射：

```text
Pixiv → jp_to_danbooru_tags.json → Final Prompt
```

未知 Tags：

```text
Pixiv unknown
→ 本地 LLM 生成 Danbooru-style candidate
→ NAIDv3 Tag Search exact match (#)
→ n_count / Power
```

当前自动加入策略：

```text
NAID Tag Suggest exact match
→ 自动加入

Danbooru fallback / unknown / lookup unavailable
→ 不自动加入，只保留为 suggested/unverified
```

NAID 查询会保存在本地 cache，避免重复请求外部免费站点。

## 6. 数据与隐私

`.gitignore` 默认排除了：

- SQLite 数据库
- 上传图片
- 生成图片
- Analysis Runs
- NAID cache
- UI 状态
- LoRA / checkpoint / GGUF 等模型权重
- 本地 `generation.json`
- 本地 `anime.json`

所以 GitHub 仓库应该只保存 **代码、公开配置模板与非私人 tag mapping**。

提交前仍建议运行：

```powershell
git status
```

确认没有私人文件被 staging。

## 7. GitHub 初始化

在项目目录：

```powershell
git init
git add .
git commit -m "Initial Illustrious Reconstruction Studio v0.5.0"
git branch -M main
```

然后在 GitHub 创建一个空仓库，例如：

```text
illustrious-reconstruction-studio
```

再执行 GitHub 给出的：

```powershell
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

建议一开始建 **Private repository**。确认没有私人内容后，再决定是否公开。

## 8. 数据架构

核心关系：

```text
Image
 └─ Analysis Run(s)
      ├─ Source provenance
      ├─ WD14
      ├─ VLM
      └─ Final Prompt
           └─ Generation Run(s)
                ├─ Seed
                ├─ LoRA recipe
                ├─ Workflow snapshot
                ├─ Generated image
                ├─ Prompt correction provenance
                └─ Rating / comment
```

同一张图重新分析不会覆盖旧 Analysis Run。

同一 Prompt 也可以生成多个不同 seed，并分别评分。

Analyzer v2.5.1 会在 run JSON / SQLite `raw_json` 中分别保留：

```text
visual_features_model_raw       = merger 模型提取结果
visual_features_deterministic   = 从 VLM 原文确定性恢复的漏项（含 origin / evidence）
visual_features_raw             = 两者合并后的过滤输入
```

确定性恢复只接受 VLM 明确出现的短语，并排除 `no bokeh` 等否定语境。

## 9. A–E 生成质量诊断

在 Generation Settings 中可填写一个固定 seed，然后点击 `A–E 质量诊断`。
Studio 会顺序生成五张图：

```text
A = 当前 LoRA + 当前双 sampler
B = 关闭全部 LoRA + 当前双 sampler
C = 关闭全部 LoRA + 单 sampler（35 steps / CFG 6）
D = C + 半写实、高细节、绘画式渲染提示词
E = D + 按原图纵横比自动拟合分辨率
```

五张图严格共用 seed 和 checkpoint。A–C 共用当前 Final Prompt 和 workflow
分辨率，D 只追加固定的风格提示词，E 只在 D 的基础上按原图纵横比调整分辨率。
每条结果的
`sampling_json` 和完整 workflow snapshot 都会写入现有 `generation_runs`，不新增或
简化 provenance 表。候选卡片会显示 Diagnostic A–E 标识。

## 10. 当前下一步

推荐开发顺序：

1. 用当前 10–20 张图片重复 A–E 回归测试
2. 比较 Analyzer 动态风格词与 D/E 固定加权风格 preset 的评分
3. 评估将单 sampler、CFG 6、风格增强和原图纵横比做成生成 preset
4. 改进评分 UI（最佳候选、分项评分）
5. 统计 Prompt correction / LoRA / seed 与评分之间的关系
6. 后续再考虑参考图 conditioning、RAG / preference learning

当前远程访问维持同一局域网模式，不直接向公网暴露 Studio 端口。

## License

当前仓库默认 **All Rights Reserved**，不是开源许可证。

如果未来准备公开为开源项目，再将 `LICENSE` 换成 MIT / Apache-2.0 等。
