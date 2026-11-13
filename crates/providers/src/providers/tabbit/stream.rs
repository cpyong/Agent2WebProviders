//! Tabbit v3 房间事件 → OpenAI chat.completion.chunk。
//!
//! ── 上游事件形状（实测）─────────────────────────────────────
//! `POST /api/v3/chat/rooms/{id}/join` 的事件流，每帧是
//! `event: <type>` + `data: {"event_type":…,"payload":{…},"created_at":…}`：
//!
//! ```text
//!   snapshot_begin / snapshot_end / active_runs_snapshot / message_history  订阅快照，忽略
//!   run_queued / run_started / user_message_created                          控制帧
//!   assistant_message_started                                                控制帧
//!   event_message_chunk  payload.chunk_type=thinking → chunk_payload.reasoning_content
//!   assistant_message_delta  payload.delta（或 chunk_payload.content）       正文增量
//!   assistant_message_finished / run_completed                               结束
//!   run_failed / error                                                       流内错误
//! ```
//!
//! ── 为什么先窥一帧再发 head ─────────────────────────────────
//! 上游常用 **HTTP 200 + run_failed/error 事件**表达权益不足（492）或排队
//! 拒绝。只看 HTTP 状态会把它记成成功，账号也就不会轮换。所以这里在发
//! head 之前先读到「第一条正文或错误」；读到错误就把整次请求作为控制面
//! 错误返回（网关据此换号），读到正文才发 200 并开始逐帧转发。
//!
//! ── 工具调用：出流门控 ──────────────────────────────────────
//! 上游没有原生 `tool_calls`，模型把调用写成 `<tool_call>…</tool_call>`
//! （协议见 `tools.rs`）。这里不能「先原样吐出去再补一个 tool_calls」——
//! 客户端已经收到的正文无法撤回。所以正文走一道门控（[`Gate`]）：
//!
//! - 平时带 `HOLD_BACK` 字的尾巴延迟发出，尾巴里万一出现标记就能整段扣下；
//! - 命中硬标记（`<tool_call>` / ` ```tool_call `）就转「抑制」；软标记
//!   （` ```json ` / 裸 `{"name"`）只在还没发出任何正文时才认，避免把讲
//!   JSON 的正常回答也拦下来（见 [`SOFT_MARKERS`]）；
//! - 流结束时解析成功 → 发 `delta.tool_calls` + `finish_reason: "tool_calls"`；
//!   解析失败 → 把缓冲的正文补发出去，当普通回答收尾（**不报 5xx**）。
//!
//! 首字符是 `{` / `[` 的响应直接全程缓冲：那是「整段 JSON」形态的调用，
//! 提前发出去就等于把调用当正文吐给客户端。

use agent2api_provider_protocol as proto;
use agent2api_provider_sdk::ConversationSink;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use super::sign;
use super::tools::{self, Bridge};

/// 正文尾巴的保留字数：要盖得住最长标记（`<tool_call>` 11 字、围栏 13 字）。
const HOLD_BACK: usize = 48;

/// 一条 SSE 事件。
struct Event {
    name: String,
    data: String,
}

/// 翻译结果。
enum Action {
    Ignore,
    Chunks(Vec<Value>),
    Error(String, u16),
    Done(Vec<Value>),
}

/// 正文门控：决定哪些字能立刻发、哪些要扣下来等解析。
struct Gate {
    bridge: Bridge,
    buf: String,
    emitted: usize,
    suppressed: bool,
    decided: bool,
}

/// 硬标记：一出现就说明这是工具调用，立刻停止发正文。
/// `<tool_call>` / ` ```tool_call ` 不会是正常回答里的措辞，误判风险为零。
const HARD_MARKERS: [&str; 2] = [tools::OPEN_TAG, "```tool_call"];
/// 软标记：` ```json ` / 裸的 `{"name"`。正常回答（尤其讲配置、贴 JSON）
/// 里也会出现，所以只在**一个字都还没发出去**时才认 —— 那才是「模型整段
/// 只回了一个调用」的形态。正文中途出现的软标记不动，照常流式输出。
const SOFT_MARKERS: [&str; 3] = ["```json", "{\"name\"", "{\"tool_calls\""];

fn floor_boundary(text: &str, index: usize) -> usize {
    let mut at = index.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// 在 `text` 里找最早的标记，返回绝对字节下标。
fn marker_at(text: &str, allow_soft: bool) -> Option<usize> {
    let lower = text.to_ascii_lowercase();
    let mut markers: Vec<&str> = HARD_MARKERS.to_vec();
    if allow_soft {
        markers.extend(SOFT_MARKERS);
    }
    markers
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
}

impl Gate {
    fn new(bridge: Bridge) -> Self {
        Self {
            bridge,
            buf: String::new(),
            emitted: 0,
            suppressed: false,
            decided: false,
        }
    }

    /// 追加一段增量，返回**现在就可以当正文发出去**的片段。
    fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        if self.suppressed {
            return Vec::new();
        }
        if !self.decided {
            match self.buf.chars().find(|ch| !ch.is_whitespace()) {
                // 整段 JSON 形态：全程缓冲，结束时再判。
                Some('{') | Some('[') => {
                    self.suppressed = true;
                    return Vec::new();
                }
                Some(_) => self.decided = true,
                None => return Vec::new(),
            }
        }
        // 只在 emit 之后那一段里找：hold 窗口 ≥ 最长标记，标记不可能跨过
        // emitted 边界，所以从 emitted 起搜既正确又把开销压到 O(窗口)。
        if let Some(offset) = marker_at(&self.buf[self.emitted..], self.emitted == 0) {
            let at = self.emitted + offset;
            self.suppressed = true;
            if at > self.emitted {
                let piece = self.buf[self.emitted..at].to_string();
                self.emitted = at;
                return vec![piece];
            }
            return Vec::new();
        }
        let safe = floor_boundary(&self.buf, self.buf.len().saturating_sub(HOLD_BACK));
        if safe > self.emitted {
            let piece = self.buf[self.emitted..safe].to_string();
            self.emitted = safe;
            return vec![piece];
        }
        Vec::new()
    }

    /// 收尾：返回「还没发出去的正文」与解析出的调用。
    fn finish(&mut self) -> (Vec<String>, Vec<Value>) {
        let outcome = self.bridge.parse(&self.buf);
        if !outcome.calls.is_empty() {
            // 已经流出去的正文保留（通常是「我先看下文件」这类铺垫）。
            let leftover = if self.emitted == 0 && !outcome.content.is_empty() {
                vec![outcome.content]
            } else {
                Vec::new()
            };
            return (leftover, outcome.calls);
        }
        let rest = self.buf.get(self.emitted..).unwrap_or("").to_string();
        if rest.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (vec![rest], Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> Gate {
        let body = json!({
            "tools": [{"type": "function", "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }}]
        });
        Gate::new(Bridge::from_body(&body).expect("bridge"))
    }

    fn feed(gate: &mut Gate, parts: &[&str]) -> String {
        let mut out = String::new();
        for part in parts {
            for piece in gate.push(part) {
                out.push_str(&piece);
            }
        }
        out
    }

    #[test]
    fn plain_text_is_flushed_at_finish() {
        let mut gate = gate();
        assert_eq!(feed(&mut gate, &["你好", "，世界"]), "");
        let (leftover, calls) = gate.finish();
        assert_eq!(leftover.join(""), "你好，世界");
        assert!(calls.is_empty());
    }

    #[test]
    fn preamble_before_tag_is_streamed() {
        let mut gate = gate();
        let streamed = feed(
            &mut gate,
            &[
                "我先读一下文件。\n",
                "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}</tool_call>",
            ],
        );
        assert_eq!(streamed, "我先读一下文件。\n");
        let (leftover, calls) = gate.finish();
        assert!(leftover.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "read_file");
    }

    #[test]
    fn leading_json_buffers_whole_reply() {
        let mut gate = gate();
        let streamed = feed(
            &mut gate,
            &["{\"name\":\"read_file\",", "\"arguments\":{\"path\":\"b.txt\"}}"],
        );
        assert_eq!(streamed, "");
        let (leftover, calls) = gate.finish();
        assert!(leftover.is_empty());
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn soft_marker_false_positive_keeps_text() {
        let mut gate = gate();
        let text = "配置长这样：\n```json\n{\"name\":\"demo\"}\n```\n就是这样。";
        let streamed = feed(&mut gate, &[text]);
        let (leftover, calls) = gate.finish();
        assert!(calls.is_empty());
        assert_eq!(format!("{streamed}{}", leftover.join("")), text);
    }

    #[test]
    fn long_answer_streams_before_tool_call() {
        let mut gate = gate();
        let long = "这是一段很长的解释，用来确认门控只在保留窗口内拦截标记。".repeat(8);
        let streamed = feed(&mut gate, &[long.as_str()]);
        assert!(streamed.chars().count() > long.chars().count() - HOLD_BACK);
        assert!(streamed.chars().count() < long.chars().count());
        let (leftover, calls) = gate.finish();
        assert!(calls.is_empty());
        assert_eq!(
            format!("{streamed}{}", leftover.join("")),
            long,
            "正文不能多也不能少"
        );
    }
}

struct State {
    id: String,
    created: i64,
    model: String,
    started: bool,
    finished: bool,
    usage: Option<Value>,
    /// 有工具声明时做正文门控；没有就直通。
    gate: Option<Gate>,
}

impl State {
    fn new(model: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", sign::random_uuid().replace('-', "")),
            created: super::client::now() / 1000,
            model: model.to_string(),
            started: false,
            finished: false,
            usage: None,
            gate: None,
        }
    }

    fn with_tools(mut self, bridge: Option<Bridge>) -> Self {
        self.gate = bridge.map(Gate::new);
        self
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    }

    fn ensure_role(&mut self, out: &mut Vec<Value>) {
        if self.started {
            return;
        }
        self.started = true;
        out.push(self.chunk(json!({ "role": "assistant", "content": "" }), None));
    }

    /// 正文增量：有门控时先过闸，闸口只放安全的部分出来。
    fn content(&mut self, text: &str, out: &mut Vec<Value>) {
        if text.is_empty() {
            return;
        }
        let pieces = match self.gate.as_mut() {
            Some(gate) => gate.push(text),
            None => vec![text.to_string()],
        };
        for piece in pieces {
            if piece.is_empty() {
                continue;
            }
            self.ensure_role(out);
            out.push(self.chunk(json!({ "content": piece }), None));
        }
    }

    fn reasoning(&mut self, text: &str, out: &mut Vec<Value>) {
        if text.is_empty() {
            return;
        }
        self.ensure_role(out);
        out.push(self.chunk(json!({ "reasoning_content": text }), None));
    }

    fn finish(&mut self, reason: &str) -> Vec<Value> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut out = Vec::new();
        let mut reason = reason.to_string();
        if let Some(mut gate) = self.gate.take() {
            let (leftover, calls) = gate.finish();
            for piece in leftover {
                if piece.is_empty() {
                    continue;
                }
                self.ensure_role(&mut out);
                out.push(self.chunk(json!({ "content": piece }), None));
            }
            if !calls.is_empty() {
                self.ensure_role(&mut out);
                for (index, call) in calls.iter().enumerate() {
                    out.push(self.chunk(
                        json!({ "tool_calls": [{
                            "index": index,
                            "id": call.get("id").cloned().unwrap_or(Value::Null),
                            "type": "function",
                            "function": call.get("function").cloned().unwrap_or(Value::Null),
                        }]}),
                        None,
                    ));
                }
                reason = "tool_calls".to_string();
            }
        }
        self.ensure_role(&mut out);
        let reason = if reason.is_empty() {
            "stop".to_string()
        } else {
            reason
        };
        out.push(self.chunk(json!({}), Some(&reason)));
        if let Some(usage) = self.usage.clone() {
            out.push(json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage,
            }));
        }
        out
    }
}

fn text_of(value: &Value, keys: &[&str]) -> String {
    keys.iter()
        .filter_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .find(|text| !text.is_empty())
        .unwrap_or("")
        .to_string()
}

fn usage_of(value: &Value) -> Option<Value> {
    let source = value
        .get("token_usage")
        .filter(|nested| nested.is_object())
        .unwrap_or(value);
    let prompt = ["input_tokens", "prompt_tokens"]
        .iter()
        .find_map(|key| super::client::as_i64(source.get(*key).unwrap_or(&Value::Null)));
    let completion = ["output_tokens", "completion_tokens"]
        .iter()
        .find_map(|key| super::client::as_i64(source.get(*key).unwrap_or(&Value::Null)));
    if prompt.is_none() && completion.is_none() {
        return None;
    }
    let prompt = prompt.unwrap_or(0);
    let completion = completion.unwrap_or(0);
    Some(json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }))
}

fn error_status(raw: &str, message: &str) -> u16 {
    let lower = raw.to_ascii_lowercase();
    let message_lower = message.to_ascii_lowercase();
    let all = format!("{lower}\n{message_lower}");
    if all.contains("insufficient")
        || all.contains("no_quota")
        || all.contains("quota")
        || all.contains("积分不足")
        || all.contains("余额不足")
        || all.contains("额度")
        || all.contains("premium")
        || all.contains("upgrade")
        || all.contains("492")
    {
        return 429;
    }
    if all.contains("rate limit") || all.contains("too many") || all.contains("429") {
        return 429;
    }
    if all.contains("unauthorized") || all.contains("401") || all.contains("token") {
        return 401;
    }
    if all.contains("session is processing")
        || all.contains("正在处理")
        || all.contains("处理中")
        || all.contains("409")
    {
        return 409;
    }
    if all.contains("serviceavailable") && all.contains("false") {
        return 503;
    }
    502
}

fn chunks_or_idle(out: Vec<Value>) -> Action {
    // 空产出当「忽略」而不是「有正文」：否则首帧窥探会在还没读到任何字时
    // 就退出，把紧随其后的错误帧当正文流出去。
    if out.is_empty() {
        Action::Ignore
    } else {
        Action::Chunks(out)
    }
}

fn translate(event: &Event, run_id: &str, state: &mut State) -> Action {
    let raw = event.data.trim();
    if raw.is_empty() || raw == "[DONE]" {
        return Action::Ignore;
    }
    let payload: Value = match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(_) => return Action::Ignore,
    };
    let mut name = event.name.trim().to_string();
    if name.is_empty() {
        name = text_of(&payload, &["type", "event", "event_type"]);
    }
    let inner = payload.get("payload").filter(|value| value.is_object());
    // 用量可能挂在顶层、payload 或 chunk_payload 任一层（v3 实测位置不稳定）。
    if let Some(usage) = usage_of(&payload) {
        state.usage = Some(usage);
    }
    if let Some(inner) = inner {
        if let Some(usage) = usage_of(inner) {
            state.usage = Some(usage);
        }
        if let Some(chunk) = inner.get("chunk_payload") {
            if let Some(usage) = usage_of(chunk) {
                state.usage = Some(usage);
            }
        }
        if let Some(run) = inner.get("run_id").and_then(Value::as_str) {
            if !run.is_empty() && !run_id.is_empty() && run != run_id {
                return Action::Ignore;
            }
        }
        match name.as_str() {
            "event_message_chunk" => {
                let kind = text_of(inner, &["chunk_type"]);
                let chunk = inner.get("chunk_payload").unwrap_or(&Value::Null);
                return match kind.as_str() {
                    "thinking" => {
                        let mut out = Vec::new();
                        state.reasoning(
                            &text_of(chunk, &["reasoning_content", "content"]),
                            &mut out,
                        );
                        chunks_or_idle(out)
                    }
                    "thinking_finished" => Action::Ignore,
                    "error" => {
                        let message = text_of(chunk, &["message", "error_message", "detail"]);
                        Action::Error(message, error_status(raw, &text_of(inner, &["message"])))
                    }
                    _ => {
                        let mut out = Vec::new();
                        state.content(&text_of(chunk, &["content"]), &mut out);
                        chunks_or_idle(out)
                    }
                };
            }
            "assistant_message_delta" => {
                let mut text = text_of(inner, &["delta"]);
                if text.is_empty() {
                    if let Some(chunk) = inner.get("chunk_payload") {
                        text = text_of(chunk, &["content"]);
                    }
                }
                let mut out = Vec::new();
                state.content(&text, &mut out);
                return chunks_or_idle(out);
            }
            "assistant_message_finished" => return Action::Ignore,
            "run_completed" => {
                if let Some(usage) = usage_of(inner) {
                    state.usage = Some(usage);
                }
                let chunks = state.finish("stop");
                return Action::Done(chunks);
            }
            "run_failed" | "run_cancelled" => {
                let message = {
                    let text = text_of(inner, &["message", "error_message", "reason", "detail"]);
                    if text.is_empty() {
                        text_of(inner, &["status"])
                    } else {
                        text
                    }
                };
                let message = if message.is_empty() {
                    "Tabbit 上游拒绝了本次请求".to_string()
                } else {
                    message
                };
                return Action::Error(message.clone(), error_status(raw, &message));
            }
            "snapshot_begin" | "snapshot_end" | "message_history" | "active_runs_snapshot"
            | "run_queued" | "run_started" | "user_message_created"
            | "assistant_message_started" | "session_title_updated" | "room_reset_required" => {
                if name == "run_queued" {
                    let flat = raw
                        .to_ascii_lowercase()
                        .replace([' ', '\t', '\r', '\n'], "");
                    if flat.contains("\"serviceavailable\":false") {
                        return Action::Error(
                            "Tabbit 上游该模型此刻不可服务（排队拒绝），请稍后重试".to_string(),
                            503,
                        );
                    }
                }
                return Action::Ignore;
            }
            _ => {}
        }
    }
    // 非 v3 的扁平事件（兼容旧房间或心跳）。
    match name.as_str() {
        "thinking" => {
            let mut out = Vec::new();
            state.reasoning(&text_of(&payload, &["reasoning_content", "content", "text"]), &mut out);
            chunks_or_idle(out)
        }
        "message_chunk" | "message" | "delta" => {
            let mut text = text_of(&payload, &["content", "text", "delta"]);
            if text.is_empty() {
                if let Some(delta) = payload.get("delta") {
                    text = text_of(delta, &["content", "text"]);
                }
            }
            let mut out = Vec::new();
            state.content(&text, &mut out);
            chunks_or_idle(out)
        }
        "usage" => {
            if let Some(usage) = usage_of(&payload) {
                state.usage = Some(usage);
            }
            Action::Ignore
        }
        "error" => {
            let message = {
                let text = text_of(&payload, &["message", "msg", "error_message", "displayTips"]);
                if text.is_empty() {
                    text_of(payload.get("error").unwrap_or(&Value::Null), &["message"])
                } else {
                    text
                }
            };
            Action::Error(message, error_status(raw, ""))
        }
        "message_finish" | "finish" | "close" => {
            let reason = text_of(&payload, &["finish_reason", "stop_reason"]);
            let chunks = state.finish(&reason);
            Action::Done(chunks)
        }
        "cancel" => {
            let chunks = state.finish("stop");
            Action::Done(chunks)
        }
        "ready" | "ttft" | "title" | "switch_model" | "message_start" | "thinking_finished"
        | "risky" => Action::Ignore,
        _ => Action::Ignore,
    }
}

async fn next_event<S, T>(
    source: &mut S,
    buffer: &mut Vec<u8>,
    eof: &mut bool,
) -> Result<Option<Event>, String>
where
    S: Stream<Item = Result<T, reqwest::Error>> + Unpin,
    T: AsRef<[u8]>,
{
    loop {
        if let Some(frame) = take_frame(buffer) {
            return Ok(parse_frame(&frame));
        }
        if *eof {
            if buffer.is_empty() {
                return Ok(None);
            }
            let frame = std::mem::take(buffer);
            return Ok(parse_frame(&frame));
        }
        match source.next().await {
            Some(Ok(chunk)) => buffer.extend_from_slice(chunk.as_ref()),
            Some(Err(_)) => return Err("Tabbit 对话流传输中断".to_string()),
            None => *eof = true,
        }
    }
}

/// 取出一个完整帧（以空行结尾）；不足一帧返回 None。
fn take_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let mut line_start = 0usize;
    let mut index = 0usize;
    while index < buffer.len() {
        if buffer[index] == b'\n' {
            let mut end = index;
            if end > line_start && buffer[end - 1] == b'\r' {
                end -= 1;
            }
            if end == line_start {
                let frame = buffer[..line_start].to_vec();
                buffer.drain(..=index);
                return Some(frame);
            }
            line_start = index + 1;
        }
        index += 1;
    }
    None
}

fn parse_frame(frame: &[u8]) -> Option<Event> {
    if frame.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(frame);
    let mut name = String::new();
    let mut data = Vec::new();
    for line in text.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("event:") {
            name = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
    }
    if name.is_empty() && data.is_empty() {
        return None;
    }
    Some(Event { name, data: data.join("\n") })
}

async fn write_chunks(sink: &mut ConversationSink<'_>, chunks: &[Value]) -> Result<(), String> {
    for chunk in chunks {
        sink.send(format!("data: {chunk}\n\n").as_bytes()).await?;
    }
    Ok(())
}

async fn write_done(sink: &mut ConversationSink<'_>) -> Result<(), String> {
    sink.send(b"data: [DONE]\n\n").await
}

fn error_frame(message: &str) -> Value {
    json!({ "error": { "message": message, "type": "upstream_error" } })
}

/// 读取上游事件流并转发。head 只在这条函数内部、确认没有预置错误之后才发。
pub async fn forward(
    response: reqwest::Response,
    run_id: String,
    model: String,
    bridge: Option<Bridge>,
    sink: &mut ConversationSink<'_>,
) -> Result<(), String> {
    let mut source = Box::pin(response.bytes_stream());
    let mut buffer: Vec<u8> = Vec::new();
    let mut eof = false;
    let mut state = State::new(&model).with_tools(bridge);

    // ── 先窥到第一条正文 / 错误 / 结束（最多 64 帧控制事件）──
    let mut ready: Vec<Value> = Vec::new();
    let mut done = false;
    let mut seen = 0usize;
    loop {
        let event = match next_event(&mut source, &mut buffer, &mut eof).await? {
            Some(event) => event,
            None => break,
        };
        seen += 1;
        match translate(&event, &run_id, &mut state) {
            Action::Ignore => {
                if seen >= 64 {
                    break;
                }
            }
            Action::Chunks(chunks) => {
                ready.extend(chunks);
                break;
            }
            Action::Done(chunks) => {
                ready.extend(chunks);
                done = true;
                break;
            }
            Action::Error(message, status) => {
                let message = if message.trim().is_empty() {
                    "Tabbit 上游拒绝了本次请求".to_string()
                } else {
                    message
                };
                return Err(crate::errors::GatewayError::with_status(i32::from(status), message)
                    .into_control_message());
            }
        }
    }

    sink.head(proto::StreamHead {
        status: 200,
        headers: [("content-type".to_string(), "text/event-stream".to_string())]
            .into_iter()
            .collect(),
        upstream: proto::UpstreamKind::Chat,
    })
    .await?;
    write_chunks(sink, &ready).await?;
    if done {
        return write_done(sink).await;
    }

    // ── 继续读剩余事件 ─────────────────────────────────────
    loop {
        let event = match next_event(&mut source, &mut buffer, &mut eof).await {
            Ok(Some(event)) => event,
            Ok(None) => {
                let chunks = state.finish("stop");
                write_chunks(sink, &chunks).await?;
                return write_done(sink).await;
            }
            Err(_) => {
                let message = "Tabbit 对话流传输中断";
                write_chunks(sink, &[error_frame(message)]).await?;
                return write_done(sink).await;
            }
        };
        match translate(&event, &run_id, &mut state) {
            Action::Ignore => {}
            Action::Chunks(chunks) => write_chunks(sink, &chunks).await?,
            Action::Done(chunks) => {
                write_chunks(sink, &chunks).await?;
                return write_done(sink).await;
            }
            Action::Error(message, _) => {
                let message = if message.trim().is_empty() {
                    "Tabbit 上游拒绝了本次请求"
                } else {
                    message.as_str()
                };
                write_chunks(sink, &[error_frame(message)]).await?;
                return write_done(sink).await;
            }
        }
    }
}

