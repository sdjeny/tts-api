# tts-api 阶段完结报告
> 日期：2026-05-11 | 分支：main | 远端：github.com/sdjeny/tts-api

---

## 一、改动清单

### 1. requirements.txt — 依赖补全
从空白起步，经 5 轮缺包报错逐步补齐：

| 类别 | 包 |
|---|---|
| 核心服务 | flask>=3.0, pyyaml>=6.0 |
| Qwen3-TTS | qwen_tts, modelscope, torch>=2.0 |
| Kokoro-82M | kokoro>=0.7 |
| misaki 核心 | addict, regex |
| misaki[en] | num2words, spacy |
| misaki[zh] | pypinyin, pypinyin-dict, jieba, ordered-set, cn2an |
| 音频处理 | soundfile>=0.12, numpy>=1.24 |

**根因**：kokoro pip 元数据未声明 misaki extras（[zh]/[en]），子依赖不会自动安装，必须手动列全。

### 2. tts_client.py — 客户端兼容改造
- `submit()` 新增 `speed: float = None` 参数（可选，最小侵入）
- `language` 默认值改为 ISO 标准 `"zh"`（原为 `"Chinese"`）

### 3. handler_kokoro.py — 服务端异常可观测
- `_worker_loop()` 启动时对 `import soundfile/numpy` 加 try/except
- import 失败时打 `log.error` 并 return，不再静默吞异常

### 4. tests/test_voice_blend.py — 混合测试重写
- 中文为主，默认 `language="zh"`
- 测试覆盖：中文单音色、等权混合、加权混合、3/4音色混合
- Test 6：speed 语速参数（0.5/1.0/1.5/2.0）
- Test 7：异常情况（空 speaker、不存在音色、空文本、speed 超范围）
- Test 8：跨语言音色（中文文本+英语音色、中英混合、三语言混合、反向跨语言）
- Test 9：边界情况（极端权重、全男声/全女声四音色混合）

### 5. tests/test_zh_voices.py — 新增中文音色基线测试
- 全部 8 个中文音色逐一测试（4 女 4 男）
- 产出：生成速度 + 文件大小

---

## 二、提交记录

| Commit | 说明 |
|---|---|
| cae15ad | fix: requirements.txt 补全 kokoro 子依赖（num2words, spacy, pypinyin, jieba） |
| 60bc004 | fix: worker 启动时捕获 ImportError 打日志而非静默崩溃 |
| 9ec7e54 | feat: tts_client 补 speed 参数 + 测试改为中文为主+跨语言音色测试 |
| 76d06f8 | fix: tts_client submit 去掉 language 默认值，改可选参数 |
| 882e6c5 | fix: tts_client submit 恢复 language='Chinese' 默认值，只加 speed 参数 |
| b5b0e5a | fix: requirements.txt 补 ordered-set（misaki[zh] 依赖） |
| b6be733 | fix: test_voice_blend submit_and_wait 恢复 language 默认 Chinese |
| a856e3d | fix: language 默认值改为 ISO 标准 zh |
| e75e22e | fix: requirements.txt 补全 misaki 全部子依赖（含 cn2an、pypinyin-dict、addict、regex） |
| a37bc45 | feat: 新增中文音色基线测试（8个中文音色逐一测试） |

---

## 三、风险项

1. **Python 版本**：3.14 生态不全（num2words/spacy 无预编译包），建议用 3.12
2. **PyPI 源**：阿里云镜像缺部分包（num2words 等），装不上时切 `-i https://pypi.org/simple/`
3. **Kokoro 模型首次加载**：`KPipeline(lang_code=)` 会从 HF cache 自动加载，首次无缓存可能较慢
4. **跨语言音色**：kokoro 支持但质量未评估，需人工听验

---

## 四、待办

- [ ] 英文 54 个音色基线测试（按需）
- [ ] 跨语言音色质量人工评估
- [ ] README.md 补全（当前仅 2 行）
- [ ] .gitignore 过于激进（排除了 *.md/LICENSE/Dockerfile 等）
