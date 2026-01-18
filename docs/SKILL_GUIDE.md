# 一文带你看懂，Skills 到底是个啥

> 基于 GLM-ASR-WebUI 项目的实战经验分享

## 前言

相信最近大家在各种地方都看到一个单词 —— **Skills**。

从 GitHub 上被疯狂 star 的仓库，到各种 AI 编程工具的热门功能，Skills 的热度在 AI 圈里持续攀升。很多人都在问：Skills 到底是个啥？跟 Prompt、MCP 有什么区别？怎么用？

本文将结合 **GLM-ASR-WebUI** 项目的实际开发经验，通俗易懂地为你解析 Skills 的本质和应用场景。

---

## 一、Skills 是什么？

### 1.1 定义

**Skills**（技能）是 Anthropic 在 Claude Code 上支持的特性，于 2025 年 10 月首次推出，同年 12 月 18 日作为开放标准发布，目前已被 Claude Code、OpenCode、Codex、Cursor、Codebuddy 等主流 AI 编程工具兼容。

简单来说，Skills 是**给 Agent 用的技能包**。

### 1.2 核心特点

| 特性 | Prompt | Skills | MCP |
|------|--------|--------|-----|
| 形式 | 单个文本文件 | 文件夹（含多个资源） | 协议/服务 |
| 生命周期 | 对话级，关闭即失效 | 持久化存储 | 持久化服务 |
| 用途 | 临时指令、当场交互 | 固化流程、可复用能力 | 外部系统连接 |
| 加载方式 | 直接注入上下文 | 渐进式按需加载 | 运行时调用 |

---

## 二、理解 Skills：一个生动的比喻

想象你在工作中带新人：

- **Agent** = 一个刚入职的实习生，聪明、理解能力强，但不懂你家的规矩
- **Prompt** = 你站在他旁边口头交代任务，一次性的、临场的指令
- **Skills** = 你给他一本公司的 SOP 手册，包含规范、脚本、模板等，他可以自己查阅
- **MCP** = 给他一张门禁卡，让他能访问公司的各个系统

### 关键设计：渐进式披露

Skills 采用**渐进式披露（Progressive Disclosure）**设计原则：

1. 先加载元信息（目录）—— 让 Agent 知道"有这个手册，适用范围是啥"
2. 需要时再加载 SKILL.md（章节）—— 完整的工作流程
3. 还不够再加载其他文件（附录）—— 参考资料和脚本

这样既保证了 Agent 能准确执行任务，又节省了宝贵的 Token 上下文。

---

## 三、Skills 的文件结构

一个标准的 Skill 是一个文件夹，包含以下内容：

```
your-skill-name/
├── SKILL.md          # 唯一必需文件
├── scripts/          # 可选：脚本文件
├── templates/        # 可选：模板文件
└── resources/        # 可选：参考资源
```

### 重要规范

- **文件夹命名**：必须使用小写字母 + 连字符（如 `hotspot-collector`）
- **SKILL.md**：唯一必需文件，结构固定

### SKILL.md 结构

```markdown
---
name: 你的skill名称
description: 简要描述该技能的功能以及何时该使用它
---

# 你的技能名称

## 指令 (Instructions)
为 Agent 提供清晰、逐步的操作指南。

## 示例 (Examples)
展示使用该技能的具体代码或操作案例。
```

### Description 编写规范

**description 是最核心的字段**，决定了 Agent 何时调用你的 Skill：

- ✅ **优秀**："处理 Excel 文件并生成报告"
- ❌ **不行**："我可以帮助你处理 Excel 文件"（第一人称）
- ❌ **不行**："你可以使用这个来处理 Excel 文件"（第二人称）

**最佳实践**：
- 始终使用第三人称
- 包含触发关键词
- 正文控制在 500 行以内

---

## 四、实战案例：GLM-ASR-WebUI 的多主题系统

让我们通过 GLM-ASR-WebUI 项目的最近一次提交，看看 Skills 如何在实际开发中发挥作用。

### 4.1 项目背景

GLM-ASR-WebUI 是一个基于 GLM-ASR 模型的语音识别 Web 服务，拥有赛博朋克风格的界面。

### 4.2 开发任务

**需求**：为 WebUI 添加多主题系统，支持 6 种风格：
- 霓虹赛博（默认）
- 极简白
- 深色专业
- 温暖日落
- 自然森林
- 深海极光

### 4.3 使用 Skills 的开发流程

**第一步：规划**

```
开发者：帮我在 GLM-ASR-WebUI 项目中添加一个多主题系统，支持 6 种主题风格。

Agent（分析后）：
1. 创建主题切换器组件
2. 定义 6 套 CSS 变量
3. 实现 localStorage 持久化
4. 修复移动端样式问题
```

**第二步：执行**

Agent 调用 `frontend-design` Skill，按照以下步骤完成开发：

1. 添加主题切换器 HTML 结构
2. 编写 6 套完整的主题 CSS 变量
3. 实现主题切换 JavaScript 逻辑
4. 修复赛博主题中示例音频样式缺失问题
5. 优化移动端布局

**第三步：迭代**

发现示例音频 URL 只显示相对路径，Agent 自动修复，确保正确拼接完整域名。

### 4.4 最终成果

```diff
 templates/index.html | 2511 ++++++++++++++++++++++++++++++++++++++------------
 1 file changed, 1942 insertions(+), 569 deletions(-)
```

**新增功能**：
- 6 种完整的主题风格
- 主题切换器组件
- localStorage 持久化存储
- 移动端优化布局
- 自动域名拼接修复

---

## 五、如何创建和安装 Skills

### 5.1 使用 Skill Creator

Anthropic 官方提供了 `skill-creator` Skill，可以帮你快速创建新的 Skills。

**安装命令**：

```
安装这个 skill，skill 项目地址为: https://github.com/anthropics/skills/tree/main/skills/skill-creator
```

### 5.2 手动安装

**方法一：命令安装**

在 Claude Code 或 OpenCode 中直接输入安装命令。

**方法二：文件夹复制**

将 Skill 文件夹复制到全局目录：

- **Claude Code**：`~/.claude/skills`
- **OpenCode**：`~/.config/opencode/skill`

```bash
# Mac/Linux
cp -r your-skill-name ~/.config/opencode/skill/

# Windows
xcopy your-skill-name C:\Users\你的用户名\.config\opencode\skill\
```

### 5.3 推荐的官方 Skills

从 [Anthropic/skills](https://github.com/anthropics/skills) 仓库中：

| Skill | 功能 |
|-------|------|
| `docx` | Word 文档处理 |
| `frontend-design` | 前端界面开发 |
| `pdf` | PDF 操作 |
| `skill-creator` | 创建新 Skill |
| `xlsx` | Excel 表格处理 |

---

## 六、Skills vs Prompt vs MCP：如何选择？

### 6.1 使用场景对比

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 临时调整代码风格 | Prompt | 一次性指令，无需复用 |
| 固化前端开发流程 | Skills | 可复用、结构化 |
| 调用 GitHub API | MCP | 需要外部系统权限 |
| 自动化选题系统 | Skills + Agent | 多步骤流程协作 |

### 6.2 决策树

```
是否需要外部系统权限？
├─ 是 → MCP
└─ 否
    是否需要重复使用？
    ├─ 是 → Skills
    └─ 否 → Prompt
```

---

## 七、最佳实践与注意事项

### 7.1 Skills 开发最佳实践

1. **单一职责**：每个 Skill 只做一件事
2. **清晰描述**：description 字段要明确触发条件
3. **控制长度**：SKILL.md 正文不超过 500 行
4. **提供示例**：包含具体的使用案例

### 7.2 常见陷阱

- ❌ 在 description 中使用第一/第二人称
- ❌ SKILL.md 过长导致 Token 浪费
- ❌ 文件夹命名使用大写或空格
- ❌ 缺少具体示例，Agent 难以理解

### 7.3 性能优化

- 使用渐进式披露减少 Token 消耗
- 将通用逻辑放入脚本文件
- 合理组织参考资源

---

## 八、总结

Skills 的价值在于**复用**。

今天，你可以从安装 `skill-creator` 开始，把你最常用的一个动作固化下来：
- 选题筛热点
- 报错日志转修复方案
- 链接列表转摘要
- 前端主题切换系统

当它运行起来的那一瞬间，你就会懂 Skills 的魅力。

明天你会想做第二个，后天你会想把所有流程都搬进去。

到那一步，你就进入了另一个状态 —— **自由创造的状态**。

---

## 附录：资源链接

- [Anthropic 官方 Skills 仓库](https://github.com/anthropics/skills)
- [GLM-ASR-WebUI 项目](https://github.com/yourusername/GLM-ASR-WebUI)
- [Claude Code 文档](https://docs.anthropic.com/claude-code)

---

> 用 ❤️ 和 Skills 打造，让 Agent 成为你的超级助手
