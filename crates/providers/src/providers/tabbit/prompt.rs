//! 把 OpenAI 请求摊成 Tabbit 能吃的**一段纯文本**。
//!
//! 上游 completion 只收一个 `content` 字符串：没有 messages 数组、没有 role、
//! 没有原生工具调用。所以这里：
//! 1. 按角色加标记（`[系统指令]` / `[助手]` / `[工具结果]`）让模型看得出对话结构；
//! 2. 按字符预算裁剪（实测 20000 字符成功、25000 必失败）：
//!    工具协议与系统指令优先，其余**从最近往回填**；
//! 3. 工具声明交给 [`tools::Bridge`] 渲染成协议段插在最前面；工具历史
//!    （`assistant.tool_calls` / `role:"tool"`）**反向渲染**成同样的标签文本，
//!    模型才看得到「调用 → 结果」闭环。这一步是提示词级桥接，不是原生契约。
//!
//! 图片仍然拒绝：网页 v3 runs 的图片要走上传 + references，当前适配未接，
//! 不能假装看得见。

use std::collections::BTreeMap;

use serde_json::Value;

use crate::errors::GatewayError;

use super::config::{
    HISTORY_NOTE, PROMPT_BUDGET, PROMPT_BUDGET_WITH_TOOLS, PROMPT_HEAD, PROMPT_SINGLE,
    PROMPT_TAIL, SYSTEM_BUDGET, TOOL_BUDGET, TRUNC_NOTE,
};
use super::tools::{self, Bridge};

/// 一次请求的成品：要送上游的正文 + 出流时用的工具上下文。
pub struct Prepared {
    pub text: String,
    /// 有工具声明时为 `Some`，`stream.rs` 据此决定要不要做门控与解析。
    pub bridge: Option<Bridge>,
}

fn bad(message: impl Into<String>) -> GatewayError {
    GatewayError::with_status(400, message)
}

fn message_text(content: Option<&Value>) -> Result<String, GatewayError> {
    let Some(content) = content else {
        return Ok(String::new());
    };
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.trim().to_string()),
        Value::Array(parts) => {
            let mut chunks = Vec::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
                match kind {
                    "text" | "input_text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            if !text.trim().is_empty() {
                                chunks.push(text.trim().to_string());
                            }
                        }
                    }
                    "image" | "image_url" | "input_image" => {
                        return Err(bad(
                            "Tabbit 网页当前适配只支持文本，图片输入尚未接入（请移除图片或改用支持图片的提供商）",
                        ))
                    }
                    _ => {
                        return Err(bad(format!(
                            "Tabbit 网页不支持内容块类型「{kind}」，只接受文本"
                        )))
                    }
                }
            }
            Ok(chunks.join("\n"))
        }
        _ => Err(bad("Tabbit 消息内容必须是文本或文本内容块数组")),
    }
}

/// `tool_call_id → 工具名`。历史里的 `role:"tool"` 消息只带 id，不带名字，
/// 渲染成文本时必须补上名字，否则模型不知道那是什么工具的结果。
fn call_names(messages: &[Value]) -> BTreeMap<String, String> {
    let mut names = BTreeMap::new();
    for message in messages {
        let Some(calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for call in calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
            let name = call
                .get("function")
                .and_then(|func| func.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !id.is_empty() && !name.is_empty() {
                names.insert(id.to_string(), name.to_string());
            }
        }
    }
    names
}

/// `assistant.tool_calls` → 若干 `<tool_call>…</tool_call>` 文本块。
fn render_calls(calls: &[Value]) -> String {
    let mut out = String::new();
    for call in calls {
        let name = call
            .get("function")
            .and_then(|func| func.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if name.is_empty() {
            continue;
        }
        let raw = call
            .get("function")
            .and_then(|func| func.get("arguments"))
            .cloned()
            .unwrap_or_else(|| Value::String("{}".to_string()));
        // arguments 在历史里是 JSON 字符串；能解析就还原成对象，模型看得更清楚。
        let arguments = match raw {
            Value::String(text) => serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)),
            other => other,
        };
        let payload = serde_json::json!({ "name": name, "arguments": arguments });
        out.push_str(&format!("{}{}{}", tools::OPEN_TAG, payload, tools::CLOSE_TAG));
    }
    out
}

fn render_message(
    message: &Value,
    names: &BTreeMap<String, String>,
) -> Result<Option<String>, GatewayError> {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user");
    if role == "tool" || role == "function" {
        let text = message_text(message.get("content"))?;
        let name = message
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .and_then(|id| names.get(id).cloned())
            })
            .unwrap_or_else(|| "unknown".to_string());
        let body = if text.is_empty() {
            "(工具没有返回内容)".to_string()
        } else {
            text
        };
        return Ok(Some(format!("[工具结果 {name}]\n{body}")));
    }
    let mut rendered = String::new();
    let text = message_text(message.get("content"))?;
    if !text.is_empty() {
        rendered = match role {
            "system" | "developer" => format!("[系统指令]\n{text}"),
            "assistant" => format!("[助手]\n{text}"),
            _ => text,
        };
    }
    if let Some(calls) = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|calls| !calls.is_empty())
    {
        let block = render_calls(calls);
        if !block.is_empty() {
            if rendered.is_empty() {
                rendered = format!("[助手]\n{block}");
            } else {
                rendered.push('\n');
                rendered.push_str(&block);
            }
        }
    }
    if rendered.is_empty() {
        return Ok(None);
    }
    Ok(Some(rendered))
}

fn clip_text(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_string();
    }
    let mut head = PROMPT_HEAD.min(limit.saturating_sub(200).max(1));
    if limit < 200 {
        head = (limit / 2).max(1);
    }
    let tail = PROMPT_TAIL.min(limit.saturating_sub(head + 60));
    let mut out = String::new();
    out.extend(chars.iter().take(head));
    out.push_str(TRUNC_NOTE);
    if tail > 0 {
        out.extend(chars.iter().skip(chars.len() - tail));
    }
    out
}

fn join_within_budget(systems: Vec<String>, others: Vec<String>, preamble: &str) -> String {
    let total = if preamble.is_empty() {
        PROMPT_BUDGET
    } else {
        PROMPT_BUDGET_WITH_TOOLS
    };
    let mut blocks: Vec<String> = Vec::new();
    let mut used = 0usize;
    // 工具协议必须完整地留在最前面：它是「怎么调工具」的唯一说明，
    // 被裁掉就等于让模型凭空猜格式。
    if !preamble.trim().is_empty() {
        let text = clip_text(preamble, TOOL_BUDGET);
        used += text.chars().count() + 2;
        blocks.push(text);
    }
    let system_room = SYSTEM_BUDGET.min(total.saturating_sub(used).max(800));
    let system_text = clip_text(&systems.join("\n\n"), system_room);
    if !system_text.trim().is_empty() {
        used += system_text.chars().count() + 2;
        blocks.push(system_text);
    }
    let mut kept_reversed: Vec<String> = Vec::new();
    let mut dropped = false;
    for block in others.iter().rev() {
        let block = clip_text(block, PROMPT_SINGLE);
        let length = block.chars().count();
        if used + length + 2 > total {
            dropped = true;
            continue;
        }
        kept_reversed.push(block);
        used += length + 2;
    }
    if dropped {
        blocks.push(HISTORY_NOTE.to_string());
    }
    blocks.extend(kept_reversed.into_iter().rev());
    blocks.join("\n\n")
}

/// 返回要送上游的正文，以及出流时要用的工具上下文。
pub fn build(body: &Value) -> Result<Prepared, GatewayError> {
    let bridge = Bridge::from_body(body);
    let preamble = match bridge.as_ref() {
        Some(bridge) => bridge.instruction(TOOL_BUDGET),
        None => String::new(),
    };
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("Tabbit 请求缺少 messages"))?;
    if messages.is_empty() {
        return Err(bad("Tabbit messages 不能为空"));
    }
    // 没有工具声明时，单条 user 消息直接原文发出：最省 token，也最接近正常网页对话。
    if bridge.is_none()
        && messages.len() == 1
        && messages[0].get("role").and_then(Value::as_str) == Some("user")
    {
        let text = message_text(messages[0].get("content"))?;
        if !text.trim().is_empty() {
            return Ok(Prepared { text, bridge });
        }
    }
    let names = call_names(messages);
    let mut systems = Vec::new();
    let mut others = Vec::new();
    for message in messages {
        let Some(rendered) = render_message(message, &names)? else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        if matches!(role, "system" | "developer") {
            systems.push(rendered);
        } else {
            others.push(rendered);
        }
    }
    let text = join_within_budget(systems, others, &preamble);
    if text.trim().is_empty() {
        return Err(bad("Tabbit messages 里没有可发送的文本内容"));
    }
    Ok(Prepared { text, bridge })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_body(messages: Value) -> Value {
        json!({
            "messages": messages,
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
        })
    }

    #[test]
    fn single_user_message_stays_verbatim_without_tools() {
        let prepared = build(&json!({"messages": [{"role": "user", "content": "你好"}]}))
            .expect("build");
        assert_eq!(prepared.text, "你好");
        assert!(prepared.bridge.is_none());
    }

    #[test]
    fn tool_preamble_is_prepended() {
        let prepared = build(&tool_body(json!([{"role": "user", "content": "读 a.txt"}])))
            .expect("build");
        assert!(prepared.text.starts_with("## Tool calling protocol"));
        assert!(prepared.text.contains("read_file"));
        assert!(prepared.text.ends_with("读 a.txt"));
        assert!(prepared.bridge.is_some());
    }

    #[test]
    fn tool_history_is_rendered_back_to_tags() {
        let prepared = build(&tool_body(json!([
            {"role": "user", "content": "读 a.txt"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "文件内容：hi"}
        ])))
        .expect("build");
        assert!(prepared.text.contains("<tool_call>"));
        assert!(prepared.text.contains("\"name\":\"read_file\""));
        assert!(prepared.text.contains("\"arguments\":{\"path\":\"a.txt\"}"));
        assert!(prepared.text.contains("[工具结果 read_file]"));
        assert!(prepared.text.contains("文件内容：hi"));
    }

    #[test]
    fn tool_messages_no_longer_error() {
        let prepared = build(&json!({
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "function": {"name": "run", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ]
        }));
        assert!(prepared.is_ok());
    }

    #[test]
    fn system_prompt_survives_tool_budget() {
        let prepared = build(&tool_body(json!([
            {"role": "system", "content": "你是写代码的助手"},
            {"role": "user", "content": "读 a.txt"}
        ])))
        .expect("build");
        assert!(prepared.text.contains("[系统指令]"));
        assert!(prepared.text.contains("你是写代码的助手"));
    }

    #[test]
    fn image_content_still_rejected() {
        let error = build(&json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
        ]}]}))
        .err()
        .expect("应当报错");
        assert!(error.message.contains("图片"));
    }
}
