//! 提示词级工具调用（Tabbit 网页没有原生 `tool_calls`）。
//!
//! ── 为什么要桥接而不是拒绝 ──────────────────────────────────
//! 上游 v3 的 `agent_input` 只有 text/files/refs，确实没有工具契约。但 Cursor
//! 这类 Agent 客户端**每一次**请求都带 `tools`，直接拒绝等于让 Tabbit 在
//! Agent 里完全不可用（3.5.0-beta.1 就是这个行为）。所以这里做显式桥接：
//!
//! 1. 把 `tools` 归一成 `Spec`（名字 / 说明 / 参数 schema）；
//! 2. [`Bridge::instruction`] 生成一段协议说明 + 压缩 JSON Schema，由
//!    `prompt.rs` 插到系统段最前面；
//! 3. 模型按约定输出 `<tool_call>{"name":"…","arguments":{…}}</tool_call>`；
//! 4. [`Bridge::parse`] 把这些块解析成标准 `tool_calls`，`stream.rs` 据此合成
//!    `delta.tool_calls` 与 `finish_reason: "tool_calls"`；
//! 5. 历史里的 `assistant.tool_calls` / `role:"tool"` 由 `prompt.rs` 反向渲染成
//!    同样的标签文本，模型才看得到「调用 → 结果」闭环。
//!
//! ── 宁可退化成普通回答 ──────────────────────────────────────
//! 解析不出来、名字没声明、参数不是对象 —— 一律退回「整段当正文」，绝不
//! 上报 5xx。客户端看到一段带标签的文字，也好过整轮对话失败。
//!
//! ── 为什么不用随机 marker 键 ────────────────────────────────
//! 豆包工作那家是「整段 JSON + 随机键」，因为它的请求体里本来就要塞一整段
//! JSON 历史。Tabbit 送上游的是自然语言提示词，标签形态对模型更自然，也便于
//! 与真人对话混排（先讲两句再调工具）。

use serde_json::{json, Value};

use super::sign;

/// 模型要写的开标签（`stream.rs` 出流门控也认它）。
pub const OPEN_TAG: &str = "<tool_call>";
/// 对应的闭合标签。
pub const CLOSE_TAG: &str = "</tool_call>";

/// 一条已归一的工具声明。
#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// 一次请求的工具桥上下文。
#[derive(Clone, Debug)]
pub struct Bridge {
    specs: Vec<Spec>,
    /// `tool_choice: "required"` 或指定了某个函数。
    forced: bool,
    /// `parallel_tool_calls: false`。
    parallel: bool,
}

/// 解析结果。
#[derive(Debug, Default)]
pub struct Outcome {
    /// 标签块之外的正文（可能为空串）。
    pub content: String,
    /// 还原出的标准 `tool_calls`（空表示这次是普通回答）。
    pub calls: Vec<Value>,
}

fn clip(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}

fn declared(specs: &[Spec], name: &str) -> bool {
    specs.iter().any(|spec| spec.name == name)
}

/// 从一段 JSON 里挑出工具调用对象。
///
/// 认三种形状：`{"name":…,"arguments":…}`、`{"tool_calls":[…]}`
/// 以及前者的数组。`strict` 为真时（正文内嵌 JSON 的回退路径）要求键集合
/// 干净，避免把回答里正常出现的 JSON 误判成工具调用。
fn pick_calls(value: &Value, strict: bool) -> Vec<Value> {
    match value {
        Value::Array(items) => items
            .iter()
            .flat_map(|item| pick_calls(item, strict))
            .collect(),
        Value::Object(map) => {
            if let Some(Value::Array(items)) = map.get("tool_calls") {
                if strict {
                    return Vec::new();
                }
                return items
                    .iter()
                    .flat_map(|item| pick_calls(item, false))
                    .collect();
            }
            let Some(Value::String(_)) = map.get("name") else {
                return Vec::new();
            };
            if strict {
                let allowed = map
                    .keys()
                    .all(|key| matches!(key.as_str(), "name" | "arguments"));
                if !allowed {
                    return Vec::new();
                }
            }
            vec![value.clone()]
        }
        _ => Vec::new(),
    }
}

/// 从 `{` 开始尝试解析一个 JSON 值（后面可以跟别的文本）。
fn json_prefix(text: &str) -> Option<Value> {
    let mut stream = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    stream.next().and_then(Result::ok)
}

/// 一处候选载荷：`payload` 是要解析的 JSON 文本，`start..end` 是它在原文里
/// 占的字节区间（解析成功后从正文里抠掉）。
struct Candidate {
    payload: String,
    start: usize,
    end: usize,
}

fn tag_candidates(text: &str, out: &mut Vec<Candidate>) {
    let mut cursor = 0usize;
    while let Some(found) = text[cursor..].find(OPEN_TAG) {
        let start = cursor + found;
        let inner = start + OPEN_TAG.len();
        match text[inner..].find(CLOSE_TAG) {
            Some(offset) => {
                let end = inner + offset + CLOSE_TAG.len();
                out.push(Candidate {
                    payload: text[inner..inner + offset].trim().to_string(),
                    start,
                    end,
                });
                cursor = end;
            }
            None => {
                // 没闭合：从标签到文末都当载荷，解析不出来就会退回正文。
                out.push(Candidate {
                    payload: text[inner..].trim().to_string(),
                    start,
                    end: text.len(),
                });
                break;
            }
        }
    }
}

fn fence_candidates(text: &str, out: &mut Vec<Candidate>) {
    let mut cursor = 0usize;
    while let Some(found) = text[cursor..].find("```") {
        let start = cursor + found;
        let after = start + 3;
        let Some(newline) = text[after..].find('\n') else {
            break;
        };
        let body_start = after + newline + 1;
        let Some(close) = text[body_start..].find("```") else {
            break;
        };
        let body_end = body_start + close;
        if body_end > body_start {
            out.push(Candidate {
                payload: text[body_start..body_end].trim().to_string(),
                start,
                end: body_end + 3,
            });
        }
        cursor = body_end + 3;
    }
}

/// 正文里散落的 JSON 对象（模型没写标签、直接把调用当 JSON 输出）。
fn embedded_candidates(text: &str, out: &mut Vec<Candidate>) {
    let mut tried = 0usize;
    for (index, ch) in text.char_indices() {
        if ch != '{' {
            continue;
        }
        tried += 1;
        if tried > 32 {
            break;
        }
        let Some(value) = json_prefix(&text[index..]) else {
            continue;
        };
        if pick_calls(&value, true).is_empty() {
            continue;
        }
        out.push(Candidate {
            payload: text[index..].to_string(),
            start: index,
            end: text.len(),
        });
        break;
    }
}

fn strip_ranges(text: &str, ranges: &mut [(usize, usize)]) -> String {
    ranges.sort_unstable();
    let mut out = String::new();
    let mut cursor = 0usize;
    for (start, end) in ranges.iter() {
        if *start > cursor {
            out.push_str(&text[cursor..*start]);
        }
        cursor = cursor.max(*end);
    }
    if cursor < text.len() {
        out.push_str(&text[cursor..]);
    }
    out.trim().to_string()
}

/// schema 压缩的三档：0 原样、1 说明截短、2 去掉说明。
const DROP_KEYS: [&str; 5] = ["examples", "default", "title", "$schema", "$comment"];

fn prune(value: &Value, level: u8) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, item) in map {
                if DROP_KEYS.contains(&key.as_str()) {
                    continue;
                }
                match key.as_str() {
                    "description" => {
                        if level == 0 {
                            out.insert(key.clone(), item.clone());
                        } else if level == 1 {
                            if let Some(text) = item.as_str() {
                                out.insert(key.clone(), json!(clip(text, 120)));
                            }
                        }
                    }
                    "properties" => {
                        if let Some(props) = item.as_object() {
                            let mut nested = serde_json::Map::new();
                            for (name, schema) in props {
                                nested.insert(name.clone(), prune(schema, level));
                            }
                            out.insert(key.clone(), Value::Object(nested));
                        } else {
                            out.insert(key.clone(), item.clone());
                        }
                    }
                    "items" => {
                        out.insert(key.clone(), prune(item, level));
                    }
                    "anyOf" | "oneOf" | "allOf" => {
                        if let Some(list) = item.as_array() {
                            let pruned: Vec<Value> =
                                list.iter().map(|item| prune(item, level)).collect();
                            out.insert(key.clone(), Value::Array(pruned));
                        }
                    }
                    _ => {
                        out.insert(key.clone(), item.clone());
                    }
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

impl Bridge {
    /// 请求体里有可用工具声明时返回 `Some`；否则返回 `None`（走纯聊天）。
    ///
    /// 声明写错（非 function 类型、重名、缺 name）一律**跳过**这一条而不是
    /// 报错：Agent 客户端会一次塞几十个工具，为一个坏条目毁掉整轮不值得。
    pub fn from_body(body: &Value) -> Option<Self> {
        let mut specs: Vec<Spec> = Vec::new();
        let mut push = |name: &str, description: &str, parameters: &Value| {
            let name = name.trim();
            if name.is_empty() || specs.iter().any(|spec| spec.name == name) {
                return;
            }
            specs.push(Spec {
                name: name.to_string(),
                description: description.trim().to_string(),
                parameters: parameters.clone(),
            });
        };
        if let Some(rows) = body.get("tools").and_then(Value::as_array) {
            for row in rows {
                let kind = row.get("type").and_then(Value::as_str).unwrap_or("function");
                if kind != "function" {
                    continue;
                }
                let func = row.get("function").unwrap_or(row);
                let Some(name) = func.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let description = func
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let parameters = func
                    .get("parameters")
                    .filter(|value| value.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
                push(name, description, &parameters);
            }
        }
        // 旧版 `functions` 契约：直接是 `[{name, description, parameters}]`。
        if let Some(rows) = body.get("functions").and_then(Value::as_array) {
            for row in rows {
                let Some(name) = row.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let description = row
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let parameters = row
                    .get("parameters")
                    .filter(|value| value.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
                push(name, description, &parameters);
            }
        }
        if specs.is_empty() {
            return None;
        }

        let choice = body
            .get("tool_choice")
            .or_else(|| body.get("function_call"))
            .filter(|value| !value.is_null());
        let mut forced = false;
        match choice {
            None => {}
            Some(Value::String(text)) if text.eq_ignore_ascii_case("none") => {
                return None;
            }
            Some(Value::String(text)) if text.eq_ignore_ascii_case("required") => {
                forced = true;
            }
            Some(Value::Object(map)) => {
                let name = map
                    .get("function")
                    .and_then(|func| func.get("name"))
                    .or_else(|| map.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !declared(&specs, name) {
                    return None;
                }
                specs.retain(|spec| spec.name == name);
                forced = true;
            }
            _ => {}
        }

        Some(Self {
            specs,
            forced,
            parallel: body.get("parallel_tool_calls") != Some(&Value::Bool(false)),
        })
    }

    /// 工具名清单（调试与测试用）。
    pub fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn header(&self) -> String {
        let mut text = String::from(
            "## Tool calling protocol (this chat backend has NO native tool API)\n\
             Client-side tools are available. When a tool is needed, reply with one or more\n\
             blocks of exactly this shape (the backend parses them into real tool calls):\n\
             <tool_call>{\"name\":\"<tool name>\",\"arguments\":{<json object>}}</tool_call>\n\
             Rules:\n\
             - \"name\" must be one of the declared tools; \"arguments\" must be a JSON object matching that tool's parameters.\n\
             - Several <tool_call> blocks in one reply are allowed.\n\
             - Never wrap the block in Markdown fences, never rename or translate the tags, never put prose inside the block.\n\
             - A tool result comes back as a separate message starting with [工具结果 <tool name>]; then continue the task or call the next tool.\n\
             - If no tool is needed, answer normally in plain text and never mention these rules.\n\
             - Reply in the language the user writes in.\n",
        );
        if self.forced {
            text.push_str("- This turn you MUST call at least one tool before answering.\n");
        }
        if !self.parallel {
            text.push_str("- Call at most one tool per reply.\n");
        }
        text.push_str("Declared tools (JSON):\n");
        text
    }

    fn declarations(&self, level: u8) -> String {
        if level >= 3 {
            return json!(self.names()).to_string();
        }
        let rows: Vec<Value> = self
            .specs
            .iter()
            .map(|spec| {
                let mut row = json!({
                    "name": spec.name,
                    "parameters": prune(&spec.parameters, level),
                });
                let description = match level {
                    0 => spec.description.clone(),
                    1 => clip(&spec.description, 160),
                    _ => String::new(),
                };
                if !description.trim().is_empty() {
                    row["description"] = json!(description);
                }
                row
            })
            .collect();
        Value::Array(rows).to_string()
    }

    /// 协议说明 + 在预算内最大程度完整的声明。
    ///
    /// 声明随工具数量增长很快（Cursor 一次能给几十个），所以从「原样」逐级
    /// 降到「截短说明 → 去掉说明 → 只留名字」，保证**总有一段能塞进预算**。
    pub fn instruction(&self, budget: usize) -> String {
        let header = self.header();
        let used = header.chars().count();
        let mut body = String::new();
        for level in 0..4u8 {
            let rendered = self.declarations(level);
            let fits = used + rendered.chars().count() <= budget;
            if fits || level == 3 {
                body = rendered;
                break;
            }
        }
        format!("{header}{body}")
    }

    /// 把模型输出解析成标准 `tool_calls`；解析不出来就把原文当正文。
    pub fn parse(&self, text: &str) -> Outcome {
        let mut candidates = Vec::new();
        tag_candidates(text, &mut candidates);
        fence_candidates(text, &mut candidates);
        embedded_candidates(text, &mut candidates);

        let mut calls = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for candidate in &candidates {
            // 标签块套在围栏里、或内嵌 JSON 与标签块重叠时，只认第一份。
            if ranges
                .iter()
                .any(|(start, end)| candidate.start < *end && *start < candidate.end)
            {
                continue;
            }
            let Some(value) = json_prefix(&candidate.payload) else {
                continue;
            };
            let mut normalized = Vec::new();
            for raw in pick_calls(&value, false) {
                if let Some(call) = self.normalize(&raw) {
                    normalized.push(call);
                }
            }
            if normalized.is_empty() {
                continue;
            }
            calls.extend(normalized);
            ranges.push((candidate.start, candidate.end));
        }

        if calls.is_empty() {
            return Outcome {
                content: text.trim().to_string(),
                calls,
            };
        }
        Outcome {
            content: strip_ranges(text, &mut ranges),
            calls,
        }
    }

    fn normalize(&self, raw: &Value) -> Option<Value> {
        let name = raw.get("name").and_then(Value::as_str)?.trim();
        if !declared(&self.specs, name) {
            return None;
        }
        let arguments = match raw.get("arguments") {
            Some(Value::Object(map)) => Value::Object(map.clone()).to_string(),
            Some(Value::String(text)) => {
                let parsed: Value = serde_json::from_str(text).ok()?;
                if !parsed.is_object() {
                    return None;
                }
                parsed.to_string()
            }
            Some(Value::Null) | None => "{}".to_string(),
            _ => return None,
        };
        let suffix: String = sign::random_uuid()
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .take(24)
            .collect();
        Some(json!({
            "id": format!("call_{suffix}"),
            "type": "function",
            "function": { "name": name, "arguments": arguments },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::tabbit::config::TOOL_BUDGET;

    fn body() -> Value {
        json!({
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "description": "读文件", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "路径"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
                {"type": "function", "function": {"name": "run_shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}
            ]
        })
    }

    #[test]
    fn skips_request_without_tools() {
        assert!(Bridge::from_body(&json!({"messages": []})).is_none());
        assert!(Bridge::from_body(&json!({"tools": []})).is_none());
    }

    #[test]
    fn tool_choice_none_disables_bridge() {
        let mut value = body();
        value["tool_choice"] = json!("none");
        assert!(Bridge::from_body(&value).is_none());
    }

    #[test]
    fn forced_choice_keeps_single_tool() {
        let mut value = body();
        value["tool_choice"] = json!({"type": "function", "function": {"name": "run_shell"}});
        let bridge = Bridge::from_body(&value).expect("bridge");
        assert_eq!(bridge.names(), vec!["run_shell".to_string()]);
        assert!(bridge.instruction(4000).contains("MUST call at least one tool"));
    }

    #[test]
    fn instruction_carries_protocol_and_schema() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let text = bridge.instruction(4000);
        assert!(text.contains("<tool_call>"));
        assert!(text.contains("read_file"));
        assert!(text.contains("\"path\""));
    }

    #[test]
    fn instruction_degrades_inside_budget() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let tiny = bridge.instruction(400);
        assert!(tiny.chars().count() < 1200, "超预算：{}", tiny.chars().count());
        assert!(!tiny.contains("\"properties\""), "应当降到只留名字");
    }

    #[test]
    fn parses_tag_block() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let outcome = bridge.parse(
            "我先看下文件。\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}</tool_call>",
        );
        assert_eq!(outcome.calls.len(), 1);
        assert_eq!(outcome.calls[0]["function"]["name"], "read_file");
        assert_eq!(outcome.calls[0]["function"]["arguments"], "{\"path\":\"a.txt\"}");
        assert_eq!(outcome.content, "我先看下文件。");
    }

    #[test]
    fn parses_multiple_tags_and_string_arguments() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let outcome = bridge.parse(concat!(
            "<tool_call>{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}</tool_call>\n",
            "<tool_call>{\"name\":\"run_shell\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call>"
        ));
        assert_eq!(outcome.calls.len(), 2);
        assert_eq!(outcome.content, "");
        assert_eq!(outcome.calls[1]["function"]["name"], "run_shell");
    }

    #[test]
    fn parses_fenced_block() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let outcome = bridge.parse(
            "```json\n{\"name\":\"run_shell\",\"arguments\":{\"cmd\":\"ls\"}}\n```\n",
        );
        assert_eq!(outcome.calls.len(), 1);
        assert_eq!(outcome.content, "");
    }

    #[test]
    fn parses_whole_json_reply() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let outcome = bridge.parse("{\"tool_calls\":[{\"name\":\"run_shell\",\"arguments\":{\"cmd\":\"pwd\"}}]}");
        assert_eq!(outcome.calls.len(), 1);
    }

    #[test]
    fn plain_answer_stays_text() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let outcome = bridge.parse("就是这样，没有需要调用的工具。");
        assert!(outcome.calls.is_empty());
        assert_eq!(outcome.content, "就是这样，没有需要调用的工具。");
    }

    #[test]
    fn unknown_tool_falls_back_to_text() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let text = "<tool_call>{\"name\":\"drop_database\",\"arguments\":{}}</tool_call>";
        let outcome = bridge.parse(text);
        assert!(outcome.calls.is_empty());
        assert_eq!(outcome.content, text);
    }

    #[test]
    fn code_sample_is_not_a_tool_call() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let text = "配置长这样：{\"name\":\"demo\",\"version\":\"1.0.0\"}";
        let outcome = bridge.parse(text);
        assert!(outcome.calls.is_empty());
        assert_eq!(outcome.content, text);
    }

    #[test]
    fn unclosed_tag_returns_arguments() {
        let bridge = Bridge::from_body(&body()).expect("bridge");
        let outcome = bridge.parse("<tool_call>{\"name\":\"run_shell\",\"arguments\":{\"cmd\":\"ls\"}}");
        assert_eq!(outcome.calls.len(), 1);
    }

    #[test]
    fn many_large_tools_still_fit_budget() {
        // 贴近 Agent 客户端的真实形状：一次几十个工具、每个 schema 都带说明。
        let rows: Vec<Value> = (0..30)
            .map(|index| {
                json!({"type": "function", "function": {
                    "name": format!("tool_{index}"),
                    "description": "工具说明".repeat(40),
                    "parameters": {"type": "object", "properties": {
                        "path": {"type": "string", "description": "路径说明".repeat(30)},
                        "limit": {"type": "integer", "description": "数量说明".repeat(30)},
                        "mode": {"type": "string", "enum": ["a", "b", "c"], "description": "模式说明".repeat(30)}
                    }, "required": ["path"]}
                }})
            })
            .collect();
        let bridge = Bridge::from_body(&json!({ "tools": rows })).expect("bridge");
        let text = bridge.instruction(TOOL_BUDGET);
        assert!(
            text.chars().count() <= TOOL_BUDGET,
            "超出预算：{} 字",
            text.chars().count()
        );
        assert!(
            text.contains("\"properties\""),
            "降到只剩名字就不可用了：{} 字",
            text.chars().count()
        );
        assert!(text.contains("tool_29"), "工具名不能丢");
    }
}
