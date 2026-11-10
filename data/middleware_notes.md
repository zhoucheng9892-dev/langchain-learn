# 中间件机制笔记

中间件是 LangChain 1.x 最重要的扩展点。它把"围绕模型调用的各种横切逻辑"从构造函数参数里解放出来，改成可组合的钩子。

## 六个钩子

- `before_agent`：整轮开始前，适合做鉴权、加载用户画像。
- `before_model`：每次模型调用前，适合注入上下文、裁剪历史、动态改提示词。
- `after_model`：模型返回后，适合校验输出、统计 token、触发告警。
- `after_agent`：整轮结束后，适合归档、采样评测。
- `wrap_model_call`：包住整次模型调用，可以做重试、缓存、降级。
- `wrap_tool_call`：包住单次工具执行，可以做权限拦截、参数脱敏、结果截断。

## 执行顺序

`middleware=[A, B, C]` 时，`before_*` 按 A、B、C 顺序执行，`after_*` 按 C、B、A 逆序执行，形成洋葱模型。因此限流和鉴权应放在最外层，脱敏放在中间，日志放在最内层。

## 内置中间件清单

- `SummarizationMiddleware`：历史过长时自动摘要，`trigger` 支持 messages、tokens、fraction 三种阈值。
- `HumanInTheLoopMiddleware`：工具执行前中断，等待人工 approve、edit、reject 或 respond。
- `ToolCallLimitMiddleware`：限制某个工具的调用次数，支持 run_limit 和 thread_limit。
- `ModelFallbackMiddleware`：主模型失败时自动切换备用模型。
- `ToolRetryMiddleware`：工具异常自动重试，支持指数退避。
- `ToolErrorMiddleware`：统一捕获工具异常并翻译成模型能理解的话术。
- `PIIMiddleware`：对邮箱、信用卡号、IP 等做脱敏，支持 block、redact、mask、hash 四种策略。

## 常见误区

不要把权限校验写在系统提示词里。模型可以被诱导，而 `wrap_tool_call` 里的 if 判断不会。另外，中断类中间件必须配合 checkpointer 使用，否则中断后状态无处保存，无法恢复。
