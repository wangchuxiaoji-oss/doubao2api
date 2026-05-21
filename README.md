# doubao2api

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/docker-supported-blue?logo=docker)](https://github.com/wangchuxiaoji-oss/doubao2api#docker-%E9%83%A8%E7%BD%B2%E5%8F%AF%E9%80%89)
[![GitHub Stars](https://img.shields.io/github/stars/wangchuxiaoji-oss/doubao2api?style=social)](https://github.com/wangchuxiaoji-oss/doubao2api/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wangchuxiaoji-oss/doubao2api?style=social)](https://github.com/wangchuxiaoji-oss/doubao2api/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/wangchuxiaoji-oss/doubao2api)](https://github.com/wangchuxiaoji-oss/doubao2api/issues)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/wangchuxiaoji-oss/doubao2api)](https://github.com/wangchuxiaoji-oss/doubao2api/commits/main)

逆向工程豆包（Doubao）API，提供完整可用的 Python 异步客户端和 OpenAI 兼容 REST 服务。支持多轮对话（含深度思考/联网搜索）、图片生成、视频生成、音乐生成、文件上传，以及 QR 扫码登录和自动验证码处理。

> 起初只是想给自己的 Hermes Agent（基于 DeepSeek V4 Flash）补上识图能力，结果越写越多，索性做成了完整的逆向客户端和 API 服务。

## 原理

通过 QR 扫码登录（全平台）获取 `sessionid` 等认证 Cookie，然后调用豆包内部 SSE 流式端点实现对话、图片/视频/音乐生成。

| 端点 | 协议 | 思考链 | 状态 |
|------|------|--------|------|
| `POST /samantha/chat/completion` | JSON 明文 sentEvent | **有** — `block_type=10040` + `10000` | ✅ 推荐主用 |
| `POST /alice/message/stream_call_bot` | base64 编码 payload | **无** | 旧端点，已废弃 |

- 认证: Cookie (`sessionid`, `ttwid`, `passport_csrf_token`)
- 响应: Server-Sent Events 流

## 快速开始

### 安装

```bash
# 方式一：pip 安装（推荐）
pip install git+https://github.com/wangchuxiaoji-oss/doubao2api.git

# 方式二：从源码
git clone https://github.com/wangchuxiaoji-oss/doubao2api.git
cd doubao2api
pip install -e .
```

### Docker 部署（可选）

```bash
docker build -t doubao2api .
docker run -d -p 9090:9090 -v ./. doubao_session.json:/app/.doubao_session.json doubao2api
```

### 前置条件

1. Python 3.10+
2. 已安装依赖（pip install 会自动处理）

### QR 扫码登录（推荐，跨平台）

不需要安装豆包桌面客户端：

```python
from doubao2api.qr_login import QRLogin

result = QRLogin.login_and_save(".doubao_session.json")
```

### 从 Session 文件创建客户端

```python
import asyncio
from doubao2api import DoubaoChatClient

async def main():
    client = DoubaoChatClient.from_session()
    async with client:
        result = await client.chat("你好，请介绍一下你自己")
        print(result.text)

asyncio.run(main())
```

### 流式输出

```python
async with DoubaoChatClient.from_session() as client:
    async for msg in client.chat_stream("讲个笑话"):
        if msg.is_text_chunk:
            print(msg.text, end="", flush=True)
```

### 三模式对话（快速/思考/专家）

```python
from doubao2api import DoubaoChatClient, EXTENSION_BOT_ID

async with DoubaoChatClient.from_session(bot_id=EXTENSION_BOT_ID) as client:
    # 快速模式 (need_deep_think=0)
    result = await client.chat_completion("1+1=?")

    # 思考模式 (need_deep_think=1) — 带思维链
    result = await client.chat_completion("解释量子纠缠", need_deep_think=1)
    print(f"思考: {result.thinking_text}")
    print(f"回答: {result.text}")

    # 专家模式 (need_deep_think=3) — 深度推理
    result = await client.chat_completion("证明勾股定理", need_deep_think=3)
```

### 图片上传 + 多模态对话

```python
from doubao2api import DoubaoChatClient, EXTENSION_BOT_ID

async with DoubaoChatClient.from_session(bot_id=EXTENSION_BOT_ID) as client:
    image_bytes = open("photo.png", "rb").read()
    att = await client.upload_image(image_bytes, "photo.png")
    result = await client.chat_completion(
        text="描述这张图片的内容",
        image_attachments=[att],
        need_deep_think=0,
    )
```

> **注意**: 图片功能需要使用 `EXTENSION_BOT_ID`（`7338286299411103781`）。

**通过 REST API 上传大图片（推荐，无需 base64）**：
```bash
# 先上传图片，获取 CDN URL
curl -F "file=@photo.jpg" http://localhost:9090/v1/images/upload
# -> {"url": "https://...", "key": "tos-cn-i-.../xxx.png", ...}

# 然后在聊天中直接引用 URL
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"doubao-expert","messages":[{"role":"user","content":[
    {"type":"text","text":"这是什么？"},
    {"type":"image_url","image_url":{"url":"上一步返回的url"}}
  ]}],"stream":true}'
```

### 文件上传 + 文档对话

支持 PDF、TXT、DOCX、XLSX、PPTX、CSV、Markdown、代码文件等 60+ 种格式。

```python
from doubao2api import DoubaoChatClient, UploadedFile, EXTENSION_BOT_ID

async with DoubaoChatClient.from_session(bot_id=EXTENSION_BOT_ID) as client:
    # 上传文件
    file_bytes = open("report.pdf", "rb").read()
    uploaded = await client.upload_file(file_bytes, "report.pdf")
    # uploaded: UploadedFile(uri='tos-cn-i-ik7evvg4ik/xxx.pdf', name='report.pdf', size=102400, file_type='pdf')

    # 带文件引用的对话
    result = await client.chat_completion(
        text="总结这份文档的要点",
        file_attachments=[uploaded],
    )
    print(result.text)
```

#### 多文件上传

```python
async with DoubaoChatClient.from_session(bot_id=EXTENSION_BOT_ID) as client:
    # 同时上传多个文件
    files = []
    for path in ["data.csv", "readme.md", "config.json"]:
        data = open(path, "rb").read()
        uploaded = await client.upload_file(data, path)
        files.append(uploaded)

    # 一次对话引用多个文件
    result = await client.chat_completion(
        text="对比这三个文件的内容，找出关键差异",
        file_attachments=files,
    )
```

#### 文件 + 图片混合

```python
async with DoubaoChatClient.from_session(bot_id=EXTENSION_BOT_ID) as client:
    # 同时附带文件和图片
    img_att = await client.upload_image(open("chart.png", "rb").read(), "chart.png")
    file_att = await client.upload_file(open("data.csv", "rb").read(), "data.csv")

    result = await client.chat_completion(
        text="图表和数据是否一致？",
        image_attachments=[img_att],
        file_attachments=[file_att],
    )
```

#### 流式输出 + 文件

```python
async with DoubaoChatClient.from_session(bot_id=EXTENSION_BOT_ID) as client:
    uploaded = await client.upload_file(open("paper.pdf", "rb").read(), "paper.pdf")

    async for chunk in client.chat_stream_completion(
        text="逐段翻译这篇论文",
        file_attachments=[uploaded],
        need_deep_think=1,  # 思考模式
    ):
        if chunk.thinking:
            print(f"[思考] {chunk.thinking}", end="")
        if chunk.text:
            print(chunk.text, end="", flush=True)
```

#### UploadedFile 数据结构

```python
@dataclass
class UploadedFile:
    uri: str = ""        # TOS 存储路径，如 "tos-cn-i-ik7evvg4ik/xxx.pdf"
    name: str = ""       # 原始文件名
    size: int = 0        # 文件大小（字节）
    file_type: str = ""  # 文件扩展名（不含点号）
```

#### 上传流程详解

`upload_file()` 内部自动完成以下 4 步：

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: POST /alice/resource/prepare_upload                         │
│   请求: {"tenant_id":"5","scene_id":"5","resource_type":1}          │
│   响应: service_id, upload_auth_token (AK/SK/SessionToken)          │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: GET /top/v1?Action=ApplyImageUpload&ServiceId=xxx           │
│   签名: AWS Signature V4 (使用 Step 1 的 STS 凭证)                  │
│   响应: StoreUri, UploadHosts, Auth (TOS token), SessionKey         │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: POST https://{tos_host}/upload/v1/{store_uri}               │
│   Headers: Authorization={TOS Auth}, Content-CRC32={crc32_hex}      │
│   Body: 文件二进制内容                                               │
│   响应: {"code":2000,"message":"Success","data":{"crc32":"xxx"}}    │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: POST /top/v1?Action=CommitImageUpload&ServiceId=xxx         │
│   签名: AWS Signature V4                                            │
│   请求: {"SessionKey":"..."}                                        │
│   响应: UriStatus=2000 (成功)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

关键技术点：
- `/top/v1` 是豆包对 ByteDance ImageX API 的反向代理，避免了直接调用 `imagex.volcengineapi.com` 的 PSM 条件限制
- 签名使用标准 AWS Signature V4 算法（region=`cn-north-1`, service=`imagex`）
- TOS 上传需要 `Content-CRC32` header（小写 hex，8 位）
- STS 凭证有效期约 1 小时，每次上传重新获取

#### 支持的文件格式完整列表

| 类别 | 扩展名 |
|------|--------|
| 文档 | pdf, txt, csv, docx, doc, xlsx, xls, pptx, ppt, md, mobi, epub |
| Web/标记 | html, css, xml, json, yaml, yml |
| Python | py |
| JavaScript/TypeScript | js, ts, tsx, jsx |
| Java/Kotlin | java, kt |
| C/C++ | c, cpp, h, hpp |
| Go | go, mod, sum |
| Rust | rs |
| Swift | swift |
| C# | cs, xaml |
| Ruby | rb |
| PHP | php |
| Perl | pl |
| Shell | sh, bash, bat, cmd, ps1 |
| Lua | lua |
| Dart | dart |
| Scala | scala |
| Vue | vue |
| Protocol Buffers | proto |
| Docker | dockerfile |
| 配置 | env, ini, toml, plist, feature, vbs, vmx, vbox |
| 图片 | png, jpeg, jpg, webp |

#### 文件大小限制

- 单文件最大约 50MB（受 TOS 单次上传限制）
- 超大文件建议分片或使用 URL 引用方式

#### 在 chat/completion 中引用文件的 content_block 结构

文件在消息中以 `block_type=10052` 的 attachment block 传递，attachment `type=3` 表示文件：

```json
{
  "block_type": 10052,
  "content": {
    "attachment_block": {
      "attachments": [{
        "type": 3,
        "identifier": "uuid",
        "file": {
          "uri": "tos-cn-i-ik7evvg4ik/xxx.pdf",
          "url": "",
          "file_type": 0,
          "name": "report.pdf",
          "size": 102400
        },
        "parse_state": 1,
        "review_state": 1,
        "upload_status": 1,
        "progress": 100,
        "src": ""
      }]
    },
    "pc_event_block": ""
  },
  "block_id": "uuid",
  "parent_id": "",
  "meta_info": [],
  "append_fields": []
}
```

attachment type 枚举：
| type | 含义 |
|------|------|
| 1 | 图片 |
| 3 | 文件（PDF/TXT/代码等） |

### 图片生成（文生图）

```python
async with DoubaoChatClient.from_session() as client:
    result = await client.generate_image(
        prompt="一只柴犬在樱花树下",
        ratio="16:9",  # 支持 "1:1", "16:9", "9:16"
    )
    for img in result.images:
        print(f"下载: {img.ori_url}")
```

> **水印限制**: 所有 API 返回的图片 URL 均带有水印（CDN 层面通过 ImageX tplv 模板渲染），无法绕过。

### 视频生成（文生视频）

使用 Seedance 2.0 全能视频模型，每日约 10 次免费额度。

```python
async with DoubaoChatClient.from_session() as client:
    result = await client.generate_video(
        prompt="一只柴犬在雪地里奔跑",
        ratio="16:9",
        timeout=300,
    )
    for v in result.videos:
        print(f"视频: {v.video_url}")
        print(f"时长: {v.duration}s")
```

#### 图生视频（img2video）

```python
async with DoubaoChatClient.from_session() as client:
    att = await client.upload_image(open("ref.png", "rb").read(), "ref.png")
    result = await client.generate_video(
        prompt="让画面动起来，镜头缓慢推进",
        ref_image_key=att["uri"],
        ratio="16:9",
    )
```

#### 视频参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | str | 视频描述文本（必需） |
| `ratio` | str | 宽高比：`"1:1"`, `"16:9"`, `"9:16"` |
| `camera_movement` | str | 镜头运动方式（可选） |
| `ref_image_key` | str | 参考图片 key（img2video，可选） |
| `timeout` | float | 最长等待秒数，默认 300 |

#### 异步流程

```
1. POST /samantha/chat/completion (content_type=2020)
   ↓ SSE 返回文本确认 + fin_reason.async_task.id
2. POST /samantha/chat/async/stream (body: {task_id, event_id: 0})
   ↓ SSE 长连接等待 1-3 分钟
3. 收到 content_type=2021 (SamanthaVideoGenerationOutput) → 视频 URL
```

### 音乐生成（文生音乐）

音乐生成是同步的，通常 30-60 秒内返回。返回的音频为 AAC 格式，时长约 60-140 秒。

#### 最简用法（AI 全自动）

```python
async with DoubaoChatClient.from_session() as client:
    result = await client.generate_music(prompt="一首轻快的夏日流行歌曲")
    for track in result.tracks:
        print(f"标题: {track.title}")
        print(f"时长: {track.duration}s")
        print(f"音频: {track.audio_url}")
```

#### 精细控制参数

```python
result = await client.generate_music(
    prompt="一首关于星空的浪漫民谣",
    genre="Folk",
    mood="Romantic",
    gender="Male",
)
```

#### 自定义歌词模式

```python
result = await client.generate_music(
    prompt="星空之歌",
    lyric="星光洒满夜空 我在这里等你\n银河流淌不息 照亮回家的路",
    genre="Pop",
    gender="Female",
    generation_type="custome_lyric",  # 注意拼写，API 原始值
)
```

#### 参数有效值参考

所有参数值来自 `/samantha/skill/pack` API（`skill_type=9`），为**首字母大写英文**。

**genre（音乐风格）— 11 种：**

| API value | 中文 |
|-----------|------|
| `Pop` | 流行 |
| `Rock` | 摇滚 |
| `Folk` | 民谣 |
| `Electronic` | 电音 |
| `Hip Hop/Rap` | 嘻哈 |
| `Chinese Style` | 国风 |
| `DJ` | DJ |
| `R&B/Soul` | R&B |
| `Reggae` | 雷鬼 |
| `Punk` | 朋克 |
| `Jazz` | 爵士 |

**mood（情绪）— 9 种：**

| API value | 中文 |
|-----------|------|
| `Happy` | 快乐 |
| `Chill` | 放松 |
| `Dynamic/Energetic` | 活力 |
| `Excited` | 兴奋 |
| `Sentimental/Melancholic/Lonely` | 忧郁 |
| `Inspirational/Hopeful` | 鼓舞 |
| `Sorrow/Sad` | 伤感 |
| `Nostalgic/Memory` | 怀旧 |
| `Romantic` | 浪漫 |

**gender（歌手性别）：** `Male` / `Female`

**generation_type（歌词模式）：** `AI_lyric`（AI 写词，默认） / `custome_lyric`（自定义歌词）

#### 预设模板

豆包提供 31 个预设音乐模板（通过 `/samantha/skill/pack` API 获取，`skill_type=9`）。

| # | 标题 | 风格 | 情绪 | 性别 |
|---|------|------|------|------|
| 1 | 醒来 | Rock | Chill | Male |
| 2 | 关于你 | Folk | Sentimental/Melancholic/Lonely | Female |
| 3 | 玉碎银鞍 | Chinese Style | Sorrow/Sad | Female |
| 4 | 漫漫 | R&B/Soul | Chill | Male |
| 5 | 在全宇宙与你相遇 | Rock | Chill | Male |
| 6 | 牙买加的偶遇 | Reggae | Dynamic/Energetic | Male |
| 7 | 爱的具象化 | Pop | Happy | Female |
| 8 | 戏中梦 | Chinese Style | Sentimental/Melancholic/Lonely | Female |
| 9 | 凤梨罐头 | Jazz | Nostalgic/Memory | Male |
| 10 | 沉溺的戏 | Jazz | Sentimental/Melancholic/Lonely | Male |
| 11 | 坠 | Electronic | Chill | Male |
| 12 | 追 | Electronic | Dynamic/Energetic | Female |
| 13 | 寻音 | Electronic | Dynamic/Energetic | Female |
| 14 | 就粉碎我 | Punk | Excited | Male |
| 15 | 樱花树下的秘密 | Pop | Nostalgic/Memory | Female |
| 16 | 江南闲梦 | Chinese Style | Chill | Female |
| 17 | 局外人 | Folk | Nostalgic/Memory | Female |
| 18 | 归家 | Folk | Sentimental/Melancholic/Lonely | Male |
| 19 | 格桑花的等待 | DJ | Sorrow/Sad | Male |
| 20 | 丢了月亮 | Folk | Sentimental/Melancholic/Lonely | Male |
| 21 | 夏日微风的低语 | Folk | Nostalgic/Memory | Female |
| 22 | 迟到 | Pop | Chill | Female |
| 23 | 荆棘尽头 | Pop | Happy | Female |
| 24 | 空空 | Pop | Sentimental/Melancholic/Lonely | Male |
| 25 | 梦里梦外 | Pop | Sorrow/Sad | Female |
| 26 | 思卿 | Chinese Style | Sorrow/Sad | Male |
| 27 | 白瓦 | Chinese Style | Chill | Male |
| 28 | 一镜千年 | Chinese Style | Nostalgic/Memory | Male |
| 29 | 留学生 | Hip Hop/Rap | Inspirational/Hopeful | Male |
| 30 | 她的故事 | Jazz | Chill | Female |
| 31 | 不眠夜 | Punk | Excited | Male |

<details>
<summary>点击展开：31 首模板完整歌词</summary>

**1. 醒来** — Rock / Chill / Male
```
提线木偶般扭动四肢把自己叫醒在清晨
眼神在五光十色的屏幕前聚焦又涣散
划过精心包装的视频 掉进虚幻失真的世界
你在里面嬉笑怒骂 脸孔变换
曾经大言不惭的梦想还记得吗
不如冲破这茧房跃入山川湖海的怀抱
狂奔向苍穹去寻找许巍歌里的蓝莲花
让自由肆意主宰 你也终将野蛮生长
```

**2. 关于你** — Folk / Sentimental / Female
```
你是天边揉碎的白云 轻盈啊
你是擦肩落下的细雨 醉人啊
你是高悬天空的明月 遥远啊
我在夜里寂静 你在眼底蔓延
我多想 轻握着你的手
告诉你我所有最深沉的忧愁
告诉你我的秘密我的温柔
能否允许我 再陪你走过山高水长
```

**3. 玉碎银鞍** — Chinese Style / Sorrow / Female
```
往事如烟 何日重现
燃尽大殿 天人相隔不复见
银鞍白马 断弦琵琶
此生永别玉兰花
泥沼深陷 万丈深渊
半生飘摇 恨意涛涛
刀山火海走一遭 幽冥地狱催人老
谁将故人悼
```

**4. 漫漫** — R&B/Soul / Chill / Male
```
漫漫 怎么让我遇到了你
好巧你也爱在湖边看下雨
看行人走走停停
我们算不算是心有灵犀
人生一直漫漫慢慢
漫无目的地将你装在眼底
慢慢我已离不开你
慢慢我们一起变老下去
```

**5. 在全宇宙与你相遇** — Rock / Chill / Male
```
如戏般的剧本无聊的探索
宇宙的边缘是未知的轮廓
人们的生活是否有所掩饰
灵魂之外还有无尽的渴求
浩瀚 孤寂 波光粼粼
和你互相辉映
捍卫 不朽 这样足够
还好有你懂我
```

**6. 牙买加的偶遇** — Reggae / Dynamic / Male
```
我绕着加勒比海 来到牙买加
快乐如同棕榈叶 随海风飘动
偶遇了海边老人 却没有聊天
他说不要再回头看 你也可以奔跑起来
Stand up hurry up
Run up fly up
我沿着海风飘啊飘啊 就算双脚打结也不累
我看着浪潮翻啊翻啊 感受这无尽自由时间
```

**7. 爱的具象化** — Pop / Happy / Female
```
咖啡香氤氲了晨曦的窗 你在身旁陪我翻阅着时光
细雨轻敲温柔乡 我们的心跳合奏同频的交响
午后阳光洒落 捧一束浪漫与你对坐
街灯下影子交错 爱人啊再把那情话轻讲
爱的具象化 是漫步宇宙却发现你比星辰明亮
是化身仓颉 也造不出你名字的分量
是你看宏大的意象 无法写好爱的具象
就像我爱你这件事 早已成为我生命的征象
```

**8. 戏中梦** — Chinese Style / Sentimental / Female
```
梦是虚实的界限 跨越时间一步千年
那时笔墨还未染史篇 你仍是少年
你英年魂归 我迟生千年难吊唁
骤然清醒 前尘皆散独坐镜前描妆勾面
金戈铁马吞山河 封狼居胥凯歌旋
我在台上将戏演 你唯剩史书字里行间
我唱云遮月 扮将军当年风光无限
一戏唱罢英雄事 只憾今生前尘无缘相见
```

**9. 凤梨罐头** — Jazz / Nostalgic / Male
```
我打开那瓶凤梨罐头 才发现它已过了期
尝一口过期的希望 酸苦肆意侵袭
寂静的夜里回忆如潮涌 心却在游离
从一数到五百的钟响 笑我又唏嘘
```

**10. 沉溺的戏** — Jazz / Sentimental / Male
```
我以为我能轻易触碰你
我真想和你走到时间尽头
可是啊对你来说 我只是弃之可惜
这说不清的友情或是爱情 我只能沉溺
```

**11. 坠** — Electronic / Chill / Male
```
我把灵魂锁进思念的茧
放任自己深潜你眸中的海洋
荒芜的梦中挂满喝醉的星
摇摇欲坠向你的坐标落下
```

**12. 追** — Electronic / Dynamic / Female
```
我透过月亮 看朦胧的未来
在逐梦的途中歇脚 画幅印象派
那么忘我啊 连风都听不见
月光见证我 追在梦的后面
```

**13. 寻音** — Electronic / Dynamic / Female
```
在这个潮湿的夜晚风停住不动
树林也静默 只剩我和你相拥
就一起到遥远的梦境去找寻
那回响在灵魂深处的声音
```

**14. 就粉碎我** — Punk / Excited / Male
```
每天早上一杯速溶就随便喝
再像可怜人一样嘶吼怒骂一下
夜晚在不属于我的城市喝醉游荡
被生活粉碎被一切粉碎
就粉碎我的骄傲
就粉碎我的愤怒
就粉碎我的执着
就粉碎我的热情
```

**15. 樱花树下的秘密** — Pop / Nostalgic / Female
```
阳光洒落在旧课桌旁 我偷偷看你侧脸的光
书页间藏着的小心思 和樱花一起悄悄绽放
小卖部门口故意徘徊 只为和你多一秒遇见
其实偶尔我也会吃醋 看那女孩与你肩并肩
听说你喜欢晴天和海 我默默记在心里面
那些未曾说出口的话 藏在心底成了诗篇
这粉色秘密微酸又甜 遗憾总让机会擦肩
但愿未来会有那一天 你能听见我的心弦
```

**16. 江南闲梦** — Chinese Style / Chill / Female
```
小桥流水 轻舟摇曳过柳岸
烟雨蒙蒙 青瓦白墙映江南
茶香氤氲 书卷轻翻旧时言
江南闲梦 悠悠岁月慢慢看
诗意画卷 花开水乡间
清风拂面 心事皆释然
岁月静好 笑谈人世间
在这江南里 梦回旧时年
```

**17. 局外人** — Folk / Nostalgic / Female
```
走过林立的高楼 喧闹的街道如此空旷
逆行汹涌的人潮 犹豫着前进的方向
我在这城市中漂泊 是拼搏还是挣扎
舞台上的局外人啊 是坚强还是擅长伪装
我说理想不重 只压弯了月光
道路过于宽广 难免迷失方向
我和这城市不熟 只顾得上闯荡
白天藏好迷茫 夜晚在梦里飘荡
```

**18. 归家** — Folk / Sentimental / Male
```
坐在空空的房间 陪着我的只有吉他
城市灯火明明灭灭 孤独的人走在长街
心中的家像远方的月 看得见却触不及边界
我四处张望 眼眸藏不住失望
我想有一个家 有小小的沙发和她种的花
她会抚平我所有伤疤 她会爱我不论冬夏
心中灯塔忽暗又忽烁 归家梦从未凋落
异乡霓虹照不亮寂寞 漂泊的心还渴望着落
```

**19. 格桑花的等待** — DJ / Sorrow / Male
```
我问你是谁，那心中的爱难猜
一朵格桑花，她留守在旷野外
你说爱不在，往事深情成空白
惊鸿过一瞥，原来寂寞是天籁
什么样的情欠下何种相思债
什么样的爱等到下一次花开
缘尽情难在，还会有什么期待
相遇即离开，月光洒落秋水外
```

**20. 丢了月亮** — Folk / Sentimental / Male
```
他说想要去流浪，像一匹野马
在草原奔波辗转，听晚风歌唱
也许他真能找到，这样的地方
总好过在城市的角落，孤单流放
这一生啊，埋藏着太多的遗憾
渴望光啊，却又被现实打得遍体鳞伤
这青春啊，到底是年少的轻狂
月亮它啊，终究抵不过六便士的分量
```

**21. 夏日微风的低语** — Folk / Nostalgic / Female
```
想你像树荫下的微风
轻轻吹过我心头的夏天
盼你如晨曦穿过小窗
一丝丝光芒映在旧墙
你是我日复一日的期待
是每条小路上的远方
在每个普通的时刻里
你是无尽思念的风光
```

**22. 迟到** — Pop / Chill / Female
```
床头放着闹钟 怀里抱着猫
喂食器一工作 猫就喵喵叫
可是为什么 我还是睡过头
难道神明在说 我注定迟到
相遇迟到相恋迟到 分手也迟到
追赶不上挽留不了 只剩下苦笑
我与你的happyend 总是差一点
如果抵达不了 干脆逃开你的怀抱
```

**23. 荆棘尽头** — Pop / Happy / Female
```
多少次满身污泥，多少次人海沉浮
你从我背后走来，带动温热的晚风
水雾稀薄了，空气旋转了
绝望消散了，荡气回肠了
破除了荆棘，将苦化作了甜蜜
站在我身后，为累赋格了意义
星垂低语耳边呢喃
说千万别再走丢了
```

**24. 空空** — Pop / Sentimental / Male
```
关了灯在屋中 心太空
失去所有联络 没行踪
模糊的空气里 要发疯
乌云密布的我 是黑洞
比一块冰更寂更冷
比一尾鱼更哑更聋
沉溺在自我世界 无视情感涌动
深陷于太空四季 梦境与诗中
```

**25. 梦里梦外** — Pop / Sorrow / Female
```
在梦里爱上你
所有斑驳都重生奕奕
在梦外离开你
爱的忍耐都化为灰燃
梦中阳光耀眼
醒来只见黑暗幽远
走不出望不穿
爱的语言都成为云烟
```

**26. 思卿** — Chinese Style / Sorrow / Male
```
提笔绘，浮生卷卷落画，雕琢我心扉
叹只叹，此景难入你眼，空留了伤悲
骤雨催，车马滚滚印辙，装下离人泪
却思归，自此山高水远，恐是再难回
且看那风月笑，知是情深不可往
饮浊酒一杯，在红尘染尽爱恨憔悴
又闻那落花飞，点了流水尚余味
此一生，只一生，凭回忆将卿魂牵梦追
```

**27. 白瓦** — Chinese Style / Chill / Male
```
轻踏白瓦巷 雨后晴空长
柳絮沾露珠 釉色染霜凉
古道旁桃花香 纸伞下人相望
岁月轻抚过 留下一抹妆
白瓦映月光 故事在回响
烟雨水墨里 勾勒过往
青石板街旁 谁家炊烟扬
一幕幕画卷 藏匿旧时芳华香
```

**28. 一镜千年** — Chinese Style / Nostalgic / Male
```
铜镜轻拂尘 岁月留痕深
镜中映出旧时颜 笑语盈盈似花绽
镜中影似水流年 恍若隔世梦一场
镜外人诉尽沧桑 昔日风华已成殇
精雕细琢星云纹 凤鸟欲飞蟠龙腾
镜里千秋照出历史悠悠
汉唐宋明时光转 故事长长已泛黄
你看那谁的手拂过往事幽幽
```

**29. 留学生** — Hip Hop/Rap / Inspirational / Male
```
打包20刀的一荤一素
回到尽头那间standard room
进门先和室友打个招呼
不是Sam是袋鼠和蜘蛛
没有尖叫也不会被吓哭
我早就对此熟视无睹
比起危险还是essay更痛苦
我埋头在房间极限赶due
都说留学必然经历四个阶段
兴奋焦虑疯癫和麻木感
陌生的地域，毕业的压力
连吃饱饭都那么艰难
在异国看同一轮太阳落下
其实我也很想家
还好心态和胃很抗压
留学生的信念不会轻易崩塌
```

**30. 她的故事** — Jazz / Chill / Female
```
玫瑰会奔向远方 伴着荒郊月亮
夜莺为她歌唱 南风也为她奔忙
她会始终绽放 她会给你芳香
她会一路歌唱 去她自己的乡
```

**31. 不眠夜** — Punk / Excited / Male
```
不想听的电话就按掉
不想见的人就让他走开
不想装傻就尽情发泄吧
失败的梦想就让它算了吧
我要我要我要一个不眠夜
鼓点敲击我的大脑 吞没我的疯狂
那个瞬间 我好像看到了天堂。
哪怕只能暂时逃避 我也只想逃避
```

</details>

#### 返回结果字段

`MusicGenerationResult.tracks` 列表中每个 `GeneratedMusic` 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio_url` | str | 音频下载 URL（AAC/.mp4，抖音 CDN） |
| `title` | str | AI 生成的歌曲标题 |
| `duration` | float | 时长（秒），通常 60-140s |
| `lyrics` | str | 完整歌词文本 |
| `cover_url` | str | 封面图 URL |
| `vid` | str | 抖音视频 ID（内部标识） |

## 统一 API 服务（OpenAI 兼容）

一个服务暴露所有能力，兼容 OpenAI SDK 格式。设计为在 Linux 无头服务器上长期运行。

### 部署

#### 安装依赖

```bash
pip install aiohttp fastapi pydantic uvicorn
```

#### 本地开发启动

```bash
python -m doubao2api.unified_server
# 默认监听 127.0.0.1:9090，无认证
```

#### 生产部署（Ubuntu）

```bash
DOUBAO_API_KEY=your-secret-key \
DOUBAO_HOST=0.0.0.0 \
DOUBAO_PORT=9090 \
DOUBAO_SESSION_FILE=/opt/doubao/.doubao_session.json \
DOUBAO_RPM_LIMIT=30 \
DOUBAO_KEEPALIVE_INTERVAL=7200 \
python -m doubao2api.unified_server
```

#### systemd 服务（推荐）

```ini
# /etc/systemd/system/doubao-api.service
[Unit]
Description=Doubao Chat API
After=network.target

[Service]
Type=simple
User=doubao
WorkingDirectory=/opt/doubao
Environment=DOUBAO_API_KEY=your-secret-key
Environment=DOUBAO_HOST=0.0.0.0
Environment=DOUBAO_PORT=9090
Environment=DOUBAO_SESSION_FILE=/opt/doubao/.doubao_session.json
ExecStart=/usr/bin/python3 -m doubao2api.unified_server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now doubao-api
```

#### Nginx 反向代理（可选）

```nginx
server {
    listen 443 ssl;
    server_name doubao-api.example.com;

    location / {
        proxy_pass http://127.0.0.1:9090;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;           # SSE 流式必须关闭缓冲
        proxy_read_timeout 300s;       # 视频生成需要长超时
    }
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DOUBAO_PORT` | `9090` | 监听端口 |
| `DOUBAO_HOST` | `127.0.0.1` | 监听地址（生产环境设为 `0.0.0.0`） |
| `DOUBAO_API_KEY` | (空=无认证) | Bearer token。设为 `any` 接受任意非空 key |
| `DOUBAO_RPM_LIMIT` | `50` | 每分钟请求限制（所有端点共享） |
| `DOUBAO_TIMEOUT` | `180` | 上游请求超时秒数 |
| `DOUBAO_SESSION_FILE` | `.doubao_session.json` | Cookie 持久化文件路径 |
| `DOUBAO_KEEPALIVE_INTERVAL` | `7200` | 会话保活间隔（秒），`0` 禁用 |

### 认证

所有 API 端点（除 `GET /health` 和 `GET /admin`）需要 Bearer token：

```
Authorization: Bearer your-api-key
```

- `DOUBAO_API_KEY` 未设置时：无认证，所有请求直接通过
- `DOUBAO_API_KEY=any`：接受任意非空 Bearer token
- `DOUBAO_API_KEY=sk-xxx`：仅接受完全匹配的 token

### 会话管理

**启动行为**：如果 session 文件存在，服务启动时自动加载并初始化 client。

**自动保活**：后台每 2 小时访问 `doubao.com/chat` 页面触发 `Set-Cookie` 刷新，失败时指数退避重试（30s→60s→120s→300s）。

**对话自动清理**：每次 API 调用完成后自动删除豆包侧边栏中产生的对话记录。

**首次部署 Session 获取**：

```bash
# 方式 1：通过 Dashboard 扫码（推荐）
# 访问 http://host:port/admin?key=YOUR_API_KEY → 登录 Tab → 扫码
# 登录成功后 3 秒自动跳转概览页

# 方式 2：API 扫码登录
curl -X POST http://localhost:9090/v1/session/qr-login \
  -H "Authorization: Bearer YOUR_KEY"
# → 返回 base64 QR 码图片，用豆包 App 扫码
# 轮询状态：
curl http://localhost:9090/v1/session/qr-login \
  -H "Authorization: Bearer YOUR_KEY"

# 方式 3：手动粘贴 Cookie
curl -X POST http://localhost:9090/v1/session/update \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cookie_header": "sessionid=xxx; ttwid=yyy; passport_csrf_token=zzz"}'
```

### Admin Dashboard

内置 Web 管理面板（Vue 3 单文件应用，零构建依赖）。

**访问**：`http://host:port/admin?key=YOUR_API_KEY`

| 页面 | 功能 |
|------|------|
| **概览** | Session 状态、系统配置、Cookie 详情表格、测活按钮 |
| **登录** | QR 扫码登录，成功后 3 秒自动跳转概览页 |
| **API 测试** | 选择模型发送请求，支持流式/非流式，思维链折叠展示 |
| **请求日志** | 最近 100 条请求记录（5s 自动刷新） |

- HTML 页面本身不需要认证，数据 API 需要 Bearer token
- 概览页每 10 秒自动刷新 session 状态（仅检查 Cookie 存在性）
- "测活"按钮主动发消息验证 session 有效性

### 模型列表

| 模型 ID | 类型 | need_deep_think | 说明 |
|---------|------|-----------------|------|
| `doubao` | chat | 0 | 快速模式（默认） |
| `doubao-quick` | chat | 0 | 快速模式别名 |
| `doubao-think` | chat | 1 | 思考模式（带思维链） |
| `doubao-auto` | chat | 2 | 自动模式 |
| `doubao-expert` | chat | 3 | 专家模式（深度推理） |
| `doubao-pro` | chat | 3 | 专家模式别名 |
| `doubao-image` | image | — | 图片生成 |
| `doubao-video` | video | — | 视频生成 |
| `doubao-music` | audio | — | 音乐生成 |

### 端点详细规范

#### GET /health

健康检查，无需认证。

**响应**：
```json
{"status": "ok", "service": "doubao-unified-api"}
```

#### GET /v1/models

返回所有可用模型列表。

**响应**：
```json
{
  "object": "list",
  "data": [
    {"id": "doubao", "object": "model", "owned_by": "doubao", "created": 0},
    {"id": "doubao-think", "object": "model", "owned_by": "doubao", "created": 0}
  ]
}
```

#### POST /v1/chat/completions

OpenAI 兼容的聊天补全端点。

**请求体**：
```json
{
  "model": "doubao-think",
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
  ],
  "stream": true
}
```

**多模态（图片输入）**：
```json
{
  "model": "doubao",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 否 | 模型 ID，默认 `doubao` |
| `messages` | array | 是 | 消息数组，支持 `system`/`user`/`assistant` 角色 |
| `stream` | bool | 否 | 是否流式返回，默认 `false` |
| `temperature` | float | 否 | 温度参数（保留字段，当前不影响行为） |
| `max_tokens` | int | 否 | 最大 token 数（保留字段） |

**非流式响应**：
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "doubao-think",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "回答内容",
      "reasoning_content": "思维链内容（仅思考/专家模式）"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

**流式响应**（SSE）：
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"doubao-think","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"doubao-think","choices":[{"index":0,"delta":{"reasoning_content":"思考中..."},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"doubao-think","choices":[{"index":0,"delta":{"content":"回答"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"doubao-think","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**错误码**：
| HTTP 状态 | 说明 |
|-----------|------|
| 400 | 无效模型或空消息 |
| 401 | 认证失败 |
| 429 | 速率限制 |
| 502 | 上游错误（豆包 API 异常） |

---

#### POST /v1/files

文件上传端点。上传文件后可在 `/v1/chat/completions` 中通过 `file_url` 内容类型引用。

**请求格式**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 要上传的文件 |

**curl 示例**：
```bash
# 上传 PDF 文件
curl http://localhost:9090/v1/files \
  -F "file=@document.pdf"

# 上传代码文件
curl http://localhost:9090/v1/files \
  -F "file=@main.py"

# 上传 CSV 数据
curl http://localhost:9090/v1/files \
  -F "file=@data.csv"
```

**响应**：
```json
{
  "id": "file-a54b60e268f54d1f87210c95",
  "object": "file",
  "filename": "document.pdf",
  "bytes": 102400,
  "uri": "tos-cn-i-ik7evvg4ik/3d1fe926a54849ebaa8f69943889393a.pdf",
  "file_type": "pdf",
  "purpose": "assistants"
}
```

| 响应字段 | 类型 | 说明 |
|----------|------|------|
| `id` | string | 文件唯一标识（格式：`file-{hex24}`） |
| `object` | string | 固定 `"file"` |
| `filename` | string | 原始文件名 |
| `bytes` | int | 文件大小（字节） |
| `uri` | string | TOS 存储路径（可用于后续引用） |
| `file_type` | string | 文件扩展名 |
| `purpose` | string | 固定 `"assistants"` |

**错误码**：
| HTTP 状态 | 说明 |
|-----------|------|
| 400 | 缺少 `file` 字段或非 multipart 请求 |
| 401 | 认证失败 |
| 429 | 速率限制 |
| 502 | 上传失败（TOS 存储异常） |

---

#### GET /v1/files/download

获取已上传文件的临时 CDN 下载链接。配合 `/v1/files` 使用，实现完整的上传→下载流程。

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | `/v1/files` 返回的 TOS URI |
| `expire` | int | 否 | 期望有效期秒数（实际固定 7 天，此参数无效） |

**curl 示例**：
```bash
curl "http://localhost:9090/v1/files/download?uri=tos-cn-i-ik7evvg4ik/xxx.txt"
```

**响应**：
```json
{
  "url": "https://p3-flow-sign.byteimg.com/tos-cn-i-ik7evvg4ik/xxx.txt?x-expires=...",
  "uri": "tos-cn-i-ik7evvg4ik/xxx.txt",
  "expires_in": 3600
}
```

| 响应字段 | 类型 | 说明 |
|----------|------|------|
| `url` | string | CDN 下载链接（有效期 7 天） |
| `uri` | string | 原始 TOS URI |
| `expires_in` | int | 请求的过期时间（实际由服务端决定） |

**完整上传→下载流程**：
```bash
# 1. 上传文件
RESPONSE=$(curl -s http://localhost:9090/v1/files -F "file=@myfile.pdf")
URI=$(echo $RESPONSE | jq -r '.uri')
echo "上传成功: $URI"

# 2. 获取下载链接（可反复调用，每次返回新的 7 天有效 URL）
DOWNLOAD_URL=$(curl -s "http://localhost:9090/v1/files/download?uri=$URI" | jq -r '.url')

# 3. 下载文件
curl -o downloaded.pdf "$DOWNLOAD_URL"
```

**存储特性**：

| 项目 | 限制 |
|------|------|
| 单文件大小上限 | **1 GB** |
| 上传速度 | ~7 MB/s（取决于网络） |
| 下载 URL 有效期 | **固定 7 天**（expire 参数无效） |
| URI 持久性 | 可无限次重新获取下载 URL |
| 底层存储 | ByteDance veImageX 标准存储（默认永久保留） |
| 支持格式 | 60+ 种（PDF/DOCX/代码/图片等，见支持列表） |

> **存储持久性说明**：根据 [veImageX 官方文档](https://www.volcengine.com/docs/508/1185000)，标准存储类型的文件**默认永久保留**，不会自动删除（除非服务方配置了生命周期策略）。实测中 URI 可无限次刷新获取新的 7 天下载链接，未观察到文件被删除的情况。但由于我们使用的是豆包内部的 veImageX 服务，无法排除字节跳动未来调整内部策略的可能性，建议重要文件自行备份。

---

#### POST /v1/images/upload

图片上传端点。上传图片后返回 CDN URL，可直接用于 `/v1/chat/completions` 的 `image_url` 内容类型，**无需 base64 编码**。

适用场景：本地大图片、避免 base64 膨胀（+33% 体积）。

**请求格式**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 图片文件（png/jpg/webp） |

**curl 示例**：
```bash
# 上传图片
curl http://localhost:9090/v1/images/upload \
  -F "file=@photo.jpg"
```

**响应**：
```json
{
  "url": "https://p-vcloud.byteimg.com/tos-cn-i-ik7evvg4ik/xxx.png~tplv-...",
  "key": "tos-cn-i-ik7evvg4ik/xxx.png",
  "filename": "photo.jpg",
  "bytes": 2048576
}
```

| 响应字段 | 类型 | 说明 |
|----------|------|------|
| `url` | string | CDN URL，直接用于 `image_url.url` |
| `key` | string | TOS 存储路径 |
| `filename` | string | 原始文件名 |
| `bytes` | int | 文件大小（字节） |

**完整使用流程**：
```bash
# 1. 上传图片
URL=$(curl -s http://localhost:9090/v1/images/upload \
  -F "file=@photo.jpg" | jq -r '.url')

# 2. 在聊天中引用（无需 base64）
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"doubao-expert\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"text\", \"text\": \"描述这张图片\"},
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"$URL\"}}
    ]}],
    \"stream\": true
  }"
```

> 已上传的图片 URL（含 `tos-cn-i-`）会被自动识别，不会重复上传。

---

#### 在 /v1/chat/completions 中使用文件

有两种方式在聊天中附带文件：

**方式 1：使用 `file_url` 内容类型（自动上传）**

服务器会自动下载并上传文件到豆包存储：

```bash
# 通过 base64 data URI 传递文件内容
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "file_url", "file_url": {"url": "data:text/plain;base64,SGVsbG8gV29ybGQ="}},
        {"type": "text", "text": "这个文件里写了什么？"}
      ]
    }]
  }'

# 通过 HTTP URL 传递文件（服务器会下载）
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "file_url", "file_url": {"url": "https://example.com/report.pdf"}},
        {"type": "text", "text": "总结这份报告的要点"}
      ]
    }]
  }'
```

**方式 2：先上传再引用（适合大文件或重复引用）**

先通过 `/v1/files` 上传，然后在多次对话中复用同一个文件 URI：

```bash
# Step 1: 上传文件（只需一次）
curl -s http://localhost:9090/v1/files -F "file=@report.pdf"
# -> {"uri": "tos-cn-i-ik7evvg4ik/xxx.pdf", "filename": "report.pdf", "bytes": 102400, ...}

# Step 2: 在对话中直接引用 TOS URI（可多次复用，不会重复上传）
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-expert",
    "messages": [{"role": "user", "content": [
      {"type": "file_url", "file_url": {
        "url": "tos-cn-i-ik7evvg4ik/xxx.pdf",
        "name": "report.pdf",
        "size": 102400
      }},
      {"type": "text", "text": "总结这份报告的要点"}
    ]}]
  }'
```

**`file_url` 内容类型参数**：

```json
{
  "type": "file_url",
  "file_url": {
    "url": "data:application/pdf;base64,...",
    "name": "report.pdf",
    "size": 102400
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 文件来源，支持三种格式：<br>• `tos-cn-i-xxx/yyy.pdf` — 已上传的 TOS URI（不会重复上传）<br>• `data:{mime};base64,{data}` — 直接传递文件内容<br>• `https://example.com/file.pdf` — HTTP(S) URL，服务器会下载 |
| `name` | string | 否 | 文件名（使用 TOS URI 时建议提供） |
| `size` | int | 否 | 文件大小字节数（使用 TOS URI 时建议提供） |

**支持混合多种内容类型**：

```json
{
  "model": "doubao",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "file_url", "file_url": {"url": "data:text/csv;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "对比图表和数据是否一致"}
    ]
  }]
}
```

> **注意**：
> - 文件上传受速率限制（与聊天共享 RPM 配额）
> - 上传的文件存储在 ByteDance TOS，URI 有效期较长但非永久
> - 单次对话可附带多个文件，但总量建议不超过 5 个
> - 超大文件（>10MB）上传可能较慢，建议设置较长的请求超时

---

#### POST /v1/images/generations

图片生成端点。

**请求体**：
```json
{
  "prompt": "一只猫在月球上",
  "model": "doubao-image",
  "ratio": "16:9",
  "ref_image_url": "https://example.com/ref.png"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 图片描述 |
| `model` | string | 否 | 固定 `doubao-image` |
| `ratio` | string | 否 | 宽高比：`1:1`/`16:9`/`9:16`/`4:3`/`3:4` |
| `ref_image_url` | string | 否 | 参考图 URL 或 `data:image/png;base64,...`（图生图） |

**响应**：
```json
{
  "created": 1700000000,
  "data": [
    {
      "url": "https://p-vcloud.byteimg.com/...",
      "width": 1920,
      "height": 1080,
      "raw_url": "https://..."
    }
  ]
}
```

---

#### POST /v1/videos/generations

视频生成端点（异步）。

**请求体**：
```json
{
  "prompt": "一只柴犬在雪地奔跑",
  "ratio": "16:9",
  "camera_movement": "zoom_in",
  "stream": false
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 视频描述 |
| `model` | string | 否 | 固定 `doubao-video` |
| `ratio` | string | 否 | `1:1`/`16:9`/`9:16` |
| `camera_movement` | string | 否 | 镜头运动方式 |
| `ref_image_url` | string | 否 | 参考图（图生视频） |
| `stream` | bool | 否 | `true`=SSE 长连接等待结果，`false`=返回 task_id 轮询 |

**非流式响应**（`stream=false`，HTTP 202）：
```json
{"id": "vtask-abc123", "status": "processing", "created": 1700000000}
```

**轮询**：`GET /v1/videos/{task_id}`

```json
// 处理中
{"id": "vtask-abc123", "status": "processing"}

// 完成
{
  "id": "vtask-abc123",
  "status": "completed",
  "data": [{"url": "https://...", "cover_url": "https://...", "duration": 5.0, "width": 1920, "height": 1080}]
}

// 失败
{"id": "vtask-abc123", "status": "failed", "error": "服务过载"}
```

**流式响应**（`stream=true`，SSE）：
```
data: {"status":"completed","created":1700000000,"data":[{"url":"https://...","cover_url":"...","duration":5.0}]}

data: [DONE]
```

---

#### POST /v1/audio/generations

音乐生成端点。

**请求体**：
```json
{
  "prompt": "一首关于夏天的轻快流行歌",
  "genre": "Pop",
  "mood": "Happy",
  "gender": "Female",
  "generation_type": "AI_lyric"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 歌曲描述 |
| `model` | string | 否 | 固定 `doubao-music` |
| `lyric` | string | 否 | 自定义歌词（需配合 `generation_type: "custome_lyric"`） |
| `genre` | string | 否 | 流派（见上方参数有效值） |
| `mood` | string | 否 | 情绪 |
| `gender` | string | 否 | 声线：`Male`/`Female` |
| `theme` | string | 否 | 主题标签 |
| `generation_type` | string | 否 | `AI_lyric`/`custome_lyric`，默认 AI 写词 |

**响应**：
```json
{
  "created": 1700000000,
  "data": [
    {
      "url": "https://v3-web.douyinvod.com/...",
      "title": "夏日微风",
      "duration": 95.5,
      "lyrics": "完整歌词...",
      "cover_url": "https://..."
    }
  ]
}
```

---

#### GET /v1/session/status

查看当前 session 健康状态。

| 查询参数 | 说明 |
|----------|------|
| `probe=true` | 发送真实消息测试 session（慢，默认不启用） |

**响应**：
```json
{
  "status": "healthy",
  "age_seconds": 3600,
  "cookies_present": ["sessionid", "ttwid", "passport_csrf_token"],
  "has_sessionid": true,
  "has_csrf_token": true,
  "has_sid_guard": true
}
```

`status` 可能的值：`healthy` / `degraded` / `no_session` / `expired` / `no_client`

---

#### POST /v1/session/update

手动更新 session cookie。

**请求体**（二选一）：
```json
// 方式 1：Cookie 字典
{"cookies": {"sessionid": "xxx", "ttwid": "yyy", "passport_csrf_token": "zzz"}}

// 方式 2：Cookie 头字符串
{"cookie_header": "sessionid=xxx; ttwid=yyy; passport_csrf_token=zzz"}
```

**响应**：
```json
{"status": "ok", "message": "Session updated with 5 cookies", "cookies_received": ["sessionid", "ttwid", ...]}
```

---

#### POST /v1/session/qr-login

启动 QR 扫码登录流程。

**响应**：
```json
{
  "status": "qr_ready",
  "qr_image_base64": "iVBORw0KGgo...",
  "message": "Scan QR code with Doubao mobile app."
}
```

#### GET /v1/session/qr-login

轮询扫码状态。

**响应**：
```json
// 等待扫码
{"status": "pending"}

// 登录成功
{"status": "success", "message": "Login successful, session updated", "cookies_count": 8}

// 失败
{"status": "failed", "error": "二维码已过期"}
```

---

#### GET /admin

管理面板 HTML 页面（无需认证）。

#### GET /admin/api/logs

最近 100 条请求日志。

#### GET /admin/api/system

系统信息（Python 版本、运行时间、配置等）。

#### GET /admin/api/cookies

当前 session 的 Cookie 详情。

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9090/v1", api_key="your-key")

# 流式聊天
stream = client.chat.completions.create(
    model="doubao-think",
    messages=[{"role": "user", "content": "解释量子纠缠"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="")

# 非流式聊天
response = client.chat.completions.create(
    model="doubao-expert",
    messages=[{"role": "user", "content": "证明勾股定理"}],
)
print(response.choices[0].message.content)

# 图片生成
images = client.images.generate(
    model="doubao-image",
    prompt="一只宇航员猫咪",
    extra_body={"ratio": "1:1"},
)
print(images.data[0].url)

# 带文件的聊天（通过 base64 data URI）
import base64
file_data = open("report.pdf", "rb").read()
b64 = base64.b64encode(file_data).decode()

response = client.chat.completions.create(
    model="doubao",
    messages=[{
        "role": "user",
        "content": [
            {"type": "file_url", "file_url": {"url": f"data:application/pdf;base64,{b64}"}},
            {"type": "text", "text": "总结这份文档的核心观点"},
        ],
    }],
)
print(response.choices[0].message.content)
```

### 使用 curl

```bash
# 流式聊天
curl -N http://localhost:9090/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"doubao-think","messages":[{"role":"user","content":"你好"}],"stream":true}'

# 上传文件
curl http://localhost:9090/v1/files \
  -H "Authorization: Bearer sk-xxx" \
  -F "file=@document.pdf"

# 带文件的聊天（base64 方式）
FILE_B64=$(base64 -w0 document.pdf)
curl http://localhost:9090/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"doubao\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"file_url\",\"file_url\":{\"url\":\"data:application/pdf;base64,$FILE_B64\"}},{\"type\":\"text\",\"text\":\"总结文档\"}]}]}"

# 带文件的聊天（URL 方式，服务器自动下载）
curl http://localhost:9090/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"doubao","messages":[{"role":"user","content":[{"type":"file_url","file_url":{"url":"https://example.com/report.pdf"}},{"type":"text","text":"总结这份报告"}]}]}'

# 图片生成
curl http://localhost:9090/v1/images/generations \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"一只猫在月球上","ratio":"16:9"}'

# 视频生成（异步）
curl http://localhost:9090/v1/videos/generations \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"一只柴犬在雪地奔跑","ratio":"16:9"}'
# → {"id":"vtask-xxx","status":"processing"}

# 轮询视频状态
curl http://localhost:9090/v1/videos/vtask-xxx \
  -H "Authorization: Bearer sk-xxx"

# 音乐生成
curl http://localhost:9090/v1/audio/generations \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"一首轻快的夏日歌曲","genre":"Pop","mood":"Happy"}'
```

## Bot ID

| Bot ID | 说明 | 多模态 | 备注 |
|--------|------|--------|------|
| `7234781073513644036` | 默认豆包 AI (`DEFAULT_BOT_ID`) | 不支持 | 纯文本对话 |
| `7338286299411103781` | 扩展版 Bot (`EXTENSION_BOT_ID`) | **支持** | 图片+文本，推荐 |

> **重要**: `EXTENSION_BOT_ID` 支持图片输入（多模态），`DEFAULT_BOT_ID` 不支持。
> 统一 API 服务默认使用 `EXTENSION_BOT_ID`。

## 底层模型与路由

### 模型家族

豆包底层使用字节跳动自研的 **Seed** 大模型。三模式（快速/思考/专家）共用同一个 Seed 模型，区别在于思考/专家模式注入了 Chain-of-Thought 提示。

### seed_intention 路由表

豆包采用内容驱动的智能路由，收到消息后先分析意图再分发到对应 Agent：

| intention | detail | agent | 触发条件 |
|-----------|--------|-------|---------|
| `seed_main` | `default` | Agent-Chat | 闲聊、推理、翻译 |
| `seed_main` | `knowledge` | Agent-Knowledge | 知识问答 |
| `seed_main` | `writing` | Agent-InnerCreation | 创作 |
| `browsing` | `complex_browsing` | Agent-Knowledge | 实时搜索 |
| `multi_agent` | `Agent-Code` | Agent-Code | 代码执行 |

### 模式切换参数

| need_deep_think | 模式 | completion_option | 思考链 |
|-----------------|------|-------------------|--------|
| 0 | 快速 | `use_deep_think: false` | 无 |
| 1 | 思考 | `use_deep_think: true` | 有（10040+10000） |
| 2 | 自动 | `use_auto_cot: true` | 视内容 |
| 3 | 专家 | `use_deep_think: true, use_auto_cot: true` | 有（10040+10000） |

### API 响应中的模型元数据

SSE 事件的 `message.ext` 字段包含：

| 字段 | 含义 | 示例值 |
|------|------|--------|
| `llm_model_type` | 模型内部 ID | `38`, `1733208237` |
| `llm_intention` | 调度意图 | `seed_main` / `browsing` |
| `llm_intention_detail` | 细分意图 | `default` / `Agent-Code` |
| `input_tokens` | 输入 token 数 | 动态 |
| `output_tokens` | 输出 token 数 | 动态 |

### 火山引擎 ARK API 模型名称参考

| 系列 | model_id | 说明 |
|------|----------|------|
| 豆包 2.0 Pro | `doubao-seed-2-0-pro` | 专家模式对应 |
| 豆包 2.0 Lite | `doubao-seed-2-0-lite` | 快速模式对应 |
| 豆包 2.0 Mini | `doubao-seed-2-0-mini` | 低延迟高并发 |
| 豆包 2.0 Code | `doubao-seed-2-0-code-preview-260215` | 编程场景 |
| 豆包 1.5 Pro 32K | `doubao-1-5-pro-32k-250115` | 上一代主力 |
| 豆包 1.6 思考 | `doubao-seed-1-6-thinking-250715` | 深度推理 |
| 豆包 1.6 快速思考 | `doubao-seed-1-6-flash-250615` | 快速推理 |

## 技术细节

### 认证流程

通过 QR 扫码登录获取完整 session：

1. 请求 CSRF token（`GET https://www.doubao.com`）
2. 获取 QR 码（`POST /passport/web/scan_qrcode/`）
3. 用户用豆包 App 扫码确认
4. 获取 `sessionid`、`ttwid`、`passport_csrf_token`、`msToken` 等 Cookie
5. 持久化到 `.doubao_session.json`

### 请求格式

#### `/samantha/chat/completion`（主端点）

请求体为 JSON 明文的 `sentEvent` 对象：

```json
{
  "messages": [
    {
      "content": "{\"text\":\"你的问题\"}",
      "content_type": 2001,
      "attachments": [],
      "references": []
    }
  ],
  "completion_option": {
    "is_regen": false,
    "with_suggest": true,
    "need_create_conversation": true,
    "launch_stage": 1,
    "is_replace": false,
    "is_delete": false,
    "is_ai_playground": false,
    "memory_type": 2,
    "message_from": 0,
    "use_deep_think": true,
    "use_auto_cot": false,
    "resend_for_regen": false,
    "enable_commerce_credit": false
  },
  "evaluate_option": {"web_ab_params": ""},
  "local_conversation_id": "<timestamp>_<uuid>",
  "local_message_id": "<timestamp>_<uuid>"
}
```

关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages[].content` | string | `JSON.stringify({text: "..."})` |
| `messages[].content_type` | int | `2001` = SamanthaText |
| `completion_option.use_deep_think` | bool | 启用深度思考 |
| `completion_option.use_auto_cot` | bool | 自动 CoT |
| `completion_option.launch_stage` | int | `1` = Release |
| `completion_option.memory_type` | int | `2` = ToolMemory |

HTTP 配置：

| 项目 | 值 |
|------|-----|
| URL | `/samantha/chat/completion?aid=582478&device_platform=web&language=zh` |
| Method | POST |
| Content-Type | `application/json` |
| 额外 Header | `Agw-Js-Conv: str` |

### SSE 事件类型

| event_type | 枚举名 | 说明 |
|------------|--------|------|
| 1 | HEARTBEAT | 心跳 |
| 2001 | CMPL | 文本补全块 |
| 2002 | ACK | 消息确认（含 conversation_id） |
| 2003 | FIN | 流结束 |
| 2004 | CMD | 命令 |
| 2005 | ERR | 错误 |
| 2010 | VERBOSE | 详细元数据（seed_intention 等） |

### content_type 枚举

| 值 | 枚举名 | 说明 |
|----|--------|------|
| 2001 | SamanthaText | 普通文本 |
| 2002 | SamanthaSuggest | 推荐问题 |
| 2003 | SamanthaLoading | 加载状态（"深度思考中"） |
| 2005 | SamanthaMusicGenInput | 音乐生成输入 |
| 2008 | SamanthaSearchText | 思考链增量文本 |
| 2009 | SamanthaImageInput | 图片输入 |
| 2010 | SamanthaImageOutput | 图片输出 |
| 2020 | SamanthaVideoGenerationInput | 视频生成输入 |
| 2021 | SamanthaVideoGenerationOutput | 视频生成输出 |
| 10000 | SamanthaTextV2 | 主回答文本（思考/专家模式） |
| 10040 | BlockTypeThink | 思考内容块 |

### 思考链提取

仅 `/samantha/chat/completion` 端点可提取完整思考链：

```
SSE event_type=2001 (CMPL)
  ├─ content_type=10040 → 思考状态块（BLOCK_THINKING）
  ├─ content_type=10000 → 主回答文本
  ├─ content_type=2008  → 思考链增量文本 (content.think)
  ├─ content_type=2001  → 最终回答文本
  └─ content_type=2002  → 推荐问题
SSE event_type=2010 (VERBOSE) → 模型调度意图
SSE event_type=2002 (ACK) → conversation_id
SSE event_type=2003 (FIN) → 流结束
```

### 对话删除 API

每次 API 调用后自动删除对话：

- 端点：`POST https://www.doubao.com/samantha/thread/delete`
- Body：`{"thread_id": "<conversation_id_as_string>"}`
- 响应：`{"code": 0, "msg": ""}` (成功)
- `thread_id` 必须是字符串，整数会报 710010202

### Session 过期检测

| 端点 | 过期信号 |
|------|----------|
| `/samantha/chat/completion` | SSE `gateway-error` with `710012000` |
| `/chat/completion` | JSON `{"code": 710012001}` |

Cookie 刷新方式：`GET https://www.doubao.com/chat`（页面加载触发 `Set-Cookie`）

### msToken 与风控

`msToken` 是字节跳动前端 JSSDK（BDMS/Slardar）生成的设备指纹 token，存储在 `.bytedance.com` 域下。格式为 136 字节随机数据的 base64url 编码（184 字符）。

**服务端校验逻辑**：

| msToken 状态 | 服务端行为 |
|-------------|-----------|
| 不传（参数中不包含） | ✅ 跳过校验，正常响应 |
| 空字符串 `""` | ✅ 跳过校验，正常响应 |
| 随机伪造（格式正确但内容无效） | ❌ 触发风控 710022002 频率限制 |
| 真实有效值 | ✅ 校验通过 |

**本项目策略**：不传 msToken。QR 扫码登录无法获取 msToken（它由前端 JS 运行时生成，不参与认证流程），且服务端对"无 msToken"的容忍度远高于"假 msToken"。不传比乱传安全。

> 如果未来字节跳动加强校验（强制要求 msToken），需要通过逆向 BDMS SDK 生成有效 token，或从浏览器环境中获取。

### 搜索工具调用捕获

当豆包模型判断需要联网搜索时，SSE 流中会出现 `block_type=10025`（`search_query_result_block`）事件。本项目完整捕获这些事件并通过 API 暴露。

**SSE 流中的搜索事件结构**：

```json
{
  "block_type": 10025,
  "content": {
    "search_query_result_block": {
      "summary": "搜索 3 个关键词，参考 23 篇资料",
      "queries": ["关键词1", "关键词2", "关键词3"],
      "results": [
        {"text_card": {"title": "...", "url": "...", "summary": "...", "source_name": "..."}}
      ]
    }
  },
  "is_finish": true
}
```

**API 响应中的搜索结果**：

非流式响应在顶层增加 `search_results` 字段：

```json
{
  "id": "chatcmpl-...",
  "choices": [...],
  "search_results": {
    "type": "search",
    "summary": "搜索 3 个关键词，参考 23 篇资料",
    "queries": ["关键词1", "关键词2"],
    "is_finish": true,
    "results": [
      {"title": "...", "url": "...", "summary": "...", "source": "..."}
    ]
  }
}
```

流式响应通过 `delta.search_results` 传递（仅包含有实际结果的事件）：

```json
{"choices": [{"delta": {"role": "assistant", "search_results": {...}}}]}
```

**已知 block_type 列表**：

| block_type | 含义 | 处理方式 |
|-----------|------|---------|
| 10000 | 文本块（思维链/回答） | 拼接为 text/thinking |
| 10024 | 通用工具块（generic_tool_block） | 提取 title → tool_info |
| 10025 | 搜索结果块（search_query_result_block） | 完整解析 → search_info |
| 10040 | 思维链分隔符 | 状态机切换 thinking/answer |
| 10052 | 附件块（图片上传） | 仅用于请求构建 |
| 10101 | 加载状态块（loading_block） | 提取 text → tool_info |

## 项目结构

```
doubao2api/
├── __init__.py          # 公开 API
├── __main__.py          # CLI 入口
├── client.py            # DoubaoChatClient 主客户端
├── unified_server.py    # 统一 API 服务（OpenAI 兼容）
├── session.py           # Cookie 加载（JSON 文件）
├── sse.py               # Server-Sent Events 解析器
├── qr_login.py          # QR 扫码登录状态机
├── captcha_handler.py   # 验证码处理
├── captcha_server.py    # 验证码本地 Web 服务
└── static/
    └── admin.html       # Admin Dashboard（Vue 3）
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wangchuxiaoji-oss/doubao2api&type=Date)](https://star-history.com/#wangchuxiaoji-oss/doubao2api&Date)
