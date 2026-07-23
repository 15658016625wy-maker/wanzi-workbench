#!/usr/bin/env python3
"""
丸子的工作台 — Backend Server v2

API Endpoints:
  POST /api/ai/classify-text   — 文字快速记录分类
  POST /api/ai/classify-image  — 截图视觉识别分类
  POST /api/ai/transcribe-voice — 语音转文字+分类（预留）
  POST /api/ai/correction       — 保存纠错学习记录
  GET  /api/health              — 健康检查+模型配置信息

Security:
  - API Key 只从 .env 文件或环境变量读取，绝不暴露给前端
  - 前端只能调用本项目后端接口，不允许直接请求豆包 API
  - AI 结果必须先让用户确认，服务端只返回建议，不写入任何数据
"""

import os
import sys
import json
import base64
import io
import logging
import traceback
import re
import socket
from datetime import datetime, date
from pathlib import Path

# Load .env file before anything else
_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key and val and key not in os.environ:
            os.environ[key] = val

from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image

# ========== Configuration ==========

ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_TEXT_MODEL = os.environ.get("ARK_TEXT_MODEL", "")
ARK_VISION_MODEL = os.environ.get("ARK_VISION_MODEL", "")

DEEPSEEK_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com")

ACTIVE_PROVIDER = os.environ.get("AI_PROVIDER", "doubao")

COMPRESS_TARGET = 1 * 1024 * 1024
MAX_IMAGE_SIZE = 4 * 1024 * 1024

ALLOWED_DOMAINS = ["kindergarten", "ai", "fitness", "life", "uncertain"]
ALLOWED_TYPES = ["task", "schedule", "knowledge", "record"]
ALLOWED_PRIORITIES = ["low", "medium", "high"]

# Correction learning storage (local JSON file)
CORRECTIONS_FILE = Path(__file__).parent / "corrections.json"

# ========== Logging ==========

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wanzi-server")

# ========== Flask App ==========

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max request body

# ========== Provider Config ==========

def get_provider_config(provider: str) -> dict:
    """Return config dict for the given provider."""
    if provider == "doubao":
        api_key = ARK_API_KEY
        api_base = ARK_BASE_URL
        model_text = ARK_TEXT_MODEL
        model_vision = ARK_VISION_MODEL
        supports_json_schema = False
    elif provider == "deepseek":
        api_key = DEEPSEEK_API_KEY
        api_base = DEEPSEEK_BASE_URL
        model_text = os.environ.get("MODEL_TEXT", "deepseek-chat")
        model_vision = os.environ.get("MODEL_VISION", "deepseek-chat")
        supports_json_schema = True
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if not api_key:
        raise ValueError(
            f"API Key 未配置（{provider}）。"
            f"doubao 请设置 ARK_API_KEY，deepseek 请设置 OPENAI_API_KEY。"
            f"在项目根目录 .env 文件或环境变量中填写。"
        )
    if provider == "doubao" and not model_text:
        raise ValueError(
            "ARK_TEXT_MODEL 未配置。"
            "请前往火山方舟控制台(https://console.volcengine.com/ark) "
            "开通模型服务或创建推理接入点，将模型ID或接入点ID填入 .env 文件的 ARK_TEXT_MODEL。"
        )
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model_text": model_text,
        "model_vision": model_vision,
        "supports_json_schema": supports_json_schema,
    }


# ========== System Prompt ==========

SYSTEM_PROMPT = """你是「丸子的工作台」中的任务提取与分类助手。

你的职责不是替用户规划事情，而是从用户提供的文字、语音转写或聊天截图中，提取明确存在的信息，并提供待确认的分类建议。

核心原则：
可信优先，宁可遗漏，不得编造。

严格遵守：

1. 只能提取输入中明确出现的任务、日程、知识或记录。
2. 禁止猜测用户没有说出的意图。
3. 禁止根据常识补充准备步骤、后续步骤或关联任务。
4. 禁止把一件事项拆成多个原文没有表达的事项。
5. 每条结果必须包含 evidence。
6. evidence 必须能够在用户原始输入中直接找到。
7. evidence 不存在时，不得生成该条结果。
8. 如果没有明确任务，返回空 items。
9. 如果截图模糊、信息不完整或无法确定，返回 warning，不得猜测。
10. AI只提供建议，永远不直接保存。

分类范围：

kindergarten：
幼儿园教学、教案、幼儿、家长、家访、班级、家园沟通、教研、会议、园所、环境创设、观察记录、活动组织、台账、行政资料。

ai：
AI学习、ChatGPT、豆包、DeepSeek、Prompt、API、自动化、编程、WorkBuddy、项目开发、模型使用。

fitness：
训练、力量、跑步、瑜伽、体重、饮食、营养、增肌、减脂、健身知识、恢复。

life：
休闲、旅行、购物、护肤、医美、理财、投资、亲子兼职、路演、个人生活安排。

特别规则：
- "家访"始终优先属于 kindergarten，不得归入 life。
- "幼儿园活动""班级活动""家长活动"属于 kindergarten。
- "亲子兼职活动"只有明确属于个人兼职时才属于 life。
- 同时涉及多个领域且无法确定主要领域时，返回 uncertain。
- confidence低于0.8时必须要求用户手动确认领域。
- 不允许使用示例任务填充返回结果。

类型范围：
- task：需要执行的事项
- schedule：明确时间发生的安排
- knowledge：值得收集的知识内容
- record：已经发生、仅用于记录的事实

项目识别：
- 如果输入内容明显属于某个长期目标（如"家访""公开课""考试""健身计划"等），在 project 字段填写项目名称。
- 系统会自动匹配已有项目。如果不存在，用户可手动创建。
- 不确定时填 null。

日期处理：
- 根据 currentDate 解析今天、明天、周三、下周等相对时间。
- 无法确定具体日期时保留 null，不得猜日期。
- 没有明确时间时 time 返回 null。
- 不得擅自添加截止日期。

必须只返回符合约定结构的JSON，不得返回Markdown、解释文字或代码块。"""

JSON_FORMAT_INSTRUCTION = """

## 输出格式要求

你必须严格按照以下 JSON 格式返回结果，不要返回任何其他内容：

{
  "sourceType": "text",
  "originalText": "原文完整内容",
  "items": [
    {
      "title": "简明标题",
      "type": "task | schedule | knowledge | record",
      "domain": "kindergarten | ai | fitness | life | uncertain",
      "subcategory": "子分类名称",
      "project": "如果该任务属于某个长期目标/项目，填写项目名称；否则null",
      "date": "YYYY-MM-DD或null",
      "time": "HH:MM或null",
      "deadline": "YYYY-MM-DD或null",
      "priority": "low | medium | high 或null",
      "people": ["人名列表"],
      "evidence": "原文中支持该条记录的精确引用，必须逐字对应",
      "confidence": 0.0到1.0之间的数字
    }
  ],
  "needsConfirmation": true,
  "warnings": ["警告信息列表"]
}

注意：只返回纯JSON，不要包含任何解释文字、markdown标记或代码块标记。"""


# ========== Image Compression ==========

def compress_image(image_bytes: bytes, max_size: int = COMPRESS_TARGET) -> tuple:
    """Compress image if oversized, return (base64_str, media_type)."""
    img = Image.open(io.BytesIO(image_bytes))
    original_size = len(image_bytes)

    if img.format == "PNG":
        media_type = "image/png"
    elif img.format in ("JPEG", "JPG"):
        media_type = "image/jpeg"
    else:
        media_type = "image/jpeg"

    if original_size <= max_size:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return b64, media_type

    logger.info(f"Compressing image from {original_size} bytes (target: {max_size})")

    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg

    for quality in [85, 75, 65, 55, 45, 35]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_size:
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"

    scale = 0.75
    while scale > 0.2:
        resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=65, optimize=True)
        if buf.tell() <= max_size:
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"
        scale -= 0.15

    resized = img.resize((int(img.width * 0.2), int(img.height * 0.2)), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=35, optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"


# ========== AI Service ==========

def _build_response_format(config: dict):
    if config["supports_json_schema"]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "classify_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sourceType": {"type": "string", "enum": ["text", "wechat_image", "dingtalk_image", "other_image"]},
                        "originalText": {"type": "string"},
                        "items": {"type": "array", "items": {"type": "object"}},
                        "needsConfirmation": {"type": "boolean"},
                        "warnings": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["sourceType", "originalText", "items", "needsConfirmation", "warnings"]
                }
            }
        }
    else:
        return {"type": "json_object"}


def call_ai(text_or_image, is_image=False, source_hint="", provider=None) -> dict:
    """Unified AI call for text or image classification."""
    provider = provider or ACTIVE_PROVIDER
    config = get_provider_config(provider)
    client = OpenAI(api_key=config["api_key"], base_url=config["api_base"])

    system_content = SYSTEM_PROMPT
    if not config["supports_json_schema"]:
        system_content += JSON_FORMAT_INSTRUCTION

    # Load recent corrections as few-shot examples
    corrections = load_corrections(limit=5)
    if corrections:
        examples_text = "\n\n## 用户纠错示例（请参考这些偏好）\n"
        for c in corrections:
            examples_text += f"- 用户将 aiDomain={c['aiDomain']} 改为 {c['correctedDomain']}，aiType={c['aiType']} 改为 {c['correctedType']}\n"
        system_content += examples_text

    response_format = _build_response_format(config)

    if is_image:
        image_base64, media_type = text_or_image
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}", "detail": "high"}},
            {"type": "text", "text": f"请仔细分析这张截图中的所有文字内容，提取其中明确的任务、日程、知识或记录信息。\n{source_hint}\n注意：仅提取截图中明确出现的事项，禁止推测或补充。"}
        ]
        model = config["model_vision"]
    else:
        user_content = f"请分析以下文字内容，提取其中明确的任务、日程、知识或记录信息。\n\n原文：\n{text_or_image}"
        model = config["model_text"]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        response_format=response_format,
        temperature=0.0,
        max_tokens=4096,
        timeout=60.0 if is_image else 30.0  # Image analysis needs more time
    )

    raw = response.choices[0].message.content
    return validate_ai_response(raw, provider)


# ========== Response Validation ==========

def validate_ai_response(raw_text: str, provider: str) -> dict:
    """Parse and validate AI response with strict anti-hallucination checks."""
    # Clean up response
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned).strip()

    # Parse JSON
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                raise json.JSONDecodeError("AI返回无法解析为JSON", raw_text, 0)
        else:
            raise json.JSONDecodeError("AI返回中无JSON内容", raw_text, 0)

    # Structural validation
    if not isinstance(result, dict):
        raise ValueError("AI返回的不是JSON对象")

    result.setdefault("sourceType", "text")
    result.setdefault("originalText", "")
    result.setdefault("items", [])
    result.setdefault("needsConfirmation", True)
    result.setdefault("warnings", [])

    if not isinstance(result["items"], list):
        result["items"] = []
        result["warnings"].append("items字段不是数组，已清空")

    # Validate each item with strict anti-hallucination rules
    valid_items = []
    seen_evidence = []
    for item in result["items"]:
        if not isinstance(item, dict):
            result["warnings"].append("一条记录不是对象，已移除")
            continue

        # Rule 1: title empty → delete
        if not item.get("title", "").strip():
            result["warnings"].append("一条记录的标题为空，已移除")
            continue

        # Rule 2: evidence empty → delete
        evidence = item.get("evidence", "").strip()
        if not evidence:
            result["warnings"].append(f"「{item.get('title','')}」缺少原文依据，已移除（疑似虚构）")
            continue

        # Rule 3: domain not in allowed range → change to uncertain
        domain = item.get("domain", "")
        if domain not in ALLOWED_DOMAINS:
            item["domain"] = "uncertain"
            result["warnings"].append(f"「{item['title']}」的领域不在允许范围内，已改为uncertain")

        # Rule 4: type not in allowed range → reject
        item_type = item.get("type", "")
        if item_type not in ALLOWED_TYPES:
            result["warnings"].append(f"「{item['title']}」的类型'{item_type}'不在允许范围内，已移除")
            continue

        # Rule 5: confidence not in 0-1 → reject
        confidence = item.get("confidence", -1)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = -1
        if not (0 <= confidence <= 1):
            result["warnings"].append(f"「{item['title']}」的置信度异常({confidence})，已移除")
            continue
        item["confidence"] = confidence

        # Rule 6: priority validation
        priority = item.get("priority")
        if priority and priority not in ALLOWED_PRIORITIES:
            item["priority"] = None

        # Rule 7: ensure required fields exist
        item.setdefault("subcategory", "")
        item.setdefault("project", None)
        item.setdefault("date", None)
        item.setdefault("time", None)
        item.setdefault("deadline", None)
        item.setdefault("people", [])
        if not isinstance(item["people"], list):
            item["people"] = []

        # Rule 8: duplicate evidence → flag
        if evidence in seen_evidence:
            result["warnings"].append(f"「{item['title']}」的evidence与其他记录重复，请人工检查")
        seen_evidence.append(evidence)

        # Low confidence warning
        if confidence < 0.8:
            result["warnings"].append(f"「{item['title']}」置信度较低({confidence:.0%})，需要人工确认领域")

        valid_items.append(item)

    result["items"] = valid_items

    if any(i.get("domain") == "uncertain" for i in valid_items):
        result["warnings"].append("部分内容无法确定分类领域，请手动选择")

    result["_provider"] = provider
    config = get_provider_config(provider)
    result["_model"] = config["model_text"] if not result.get("_model") else result["_model"]

    return result


# ========== Correction Learning ==========

def load_corrections(limit=10):
    """Load recent correction records for few-shot learning."""
    if not CORRECTIONS_FILE.exists():
        return []
    try:
        data = json.loads(CORRECTIONS_FILE.read_text())
        return data[-limit:] if len(data) > limit else data
    except (json.JSONDecodeError, IOError):
        return []


def save_correction(record: dict):
    """Save a correction record."""
    corrections = load_corrections(limit=0)
    # Keep max 50 records, anonymize text
    record["createdAt"] = datetime.now().isoformat()
    # Strip full original input to protect privacy — keep only classification data
    if "originalInput" in record and len(record["originalInput"]) > 100:
        record["originalInput"] = record["originalInput"][:100] + "..."
    corrections.append(record)
    # Keep only last 50
    if len(corrections) > 50:
        corrections = corrections[-50:]
    CORRECTIONS_FILE.write_text(json.dumps(corrections, ensure_ascii=False, indent=2))


# ========== API Endpoints ==========

@app.route("/api/ai/classify-text", methods=["POST"])
def classify_text():
    """
    POST /api/ai/classify-text
    Body: {"text": "...", "currentDate": "YYYY-MM-DD", "timezone": "Asia/Shanghai"}
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "no_input", "message": "请提供 text 字段"}), 400

        current_date = data.get("currentDate", date.today().isoformat())
        timezone = data.get("timezone", "Asia/Shanghai")
        provider = data.get("provider", None)

        # Inject current date into user message for relative date resolution
        enhanced_text = f"[当前日期: {current_date}，时区: {timezone}]\n\n{text}"

        logger.info(f"classify-text: {len(text)} chars, date={current_date}, provider={provider or ACTIVE_PROVIDER}")
        result = call_ai(enhanced_text, is_image=False, provider=provider)
        result["sourceType"] = "text"
        result["originalText"] = text
        return jsonify(result)

    except ValueError as e:
        logger.error(f"Config error: {e}")
        err_msg = str(e)
        if "ARK_API_KEY" in err_msg:
            code = "ARK_KEY_MISSING"
        elif "ARK_TEXT_MODEL" in err_msg:
            code = "TEXT_MODEL_MISSING"
        elif "ARK_VISION_MODEL" in err_msg:
            code = "VISION_MODEL_MISSING"
        else:
            code = "config_error"
        return jsonify({"error": code, "message": err_msg}), 500

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return jsonify({"error": "INVALID_AI_RESPONSE", "message": "AI 返回的 JSON 格式异常，请点击重试"}), 502

    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        err = str(e)
        if "authentication" in err.lower() or "401" in err:
            return jsonify({"error": "ARK_KEY_INVALID", "message": "豆包API Key无效或已失效，请检查 .env 文件中的 ARK_API_KEY"}), 401
        elif "429" in err:
            return jsonify({"error": "rate_limit", "message": "API 调用频率超限，请稍后重试"}), 429
        elif "model" in err.lower() and ("not found" in err.lower() or "不存在" in err):
            return jsonify({"error": "MODEL_NOT_FOUND", "message": "模型ID或推理接入点不存在，请检查 .env 中的模型配置"}), 404
        elif "timeout" in err.lower() or "timed out" in err.lower():
            return jsonify({"error": "ARK_TIMEOUT", "message": "豆包响应超时，请稍后重试"}), 504
        else:
            return jsonify({"error": "api_error", "message": "AI 分析失败，请点击重试", "detail": err}), 502


@app.route("/api/ai/classify-image", methods=["POST"])
def classify_image():
    """
    POST /api/ai/classify-image
    Accepts: JSON {image: base64, sourceType, currentDate, timezone} or multipart form
    """
    try:
        current_date = date.today().isoformat()
        source_type = "other_image"
        provider = None
        timezone = "Asia/Shanghai"

        # Multipart form upload
        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()

            # Validate file type
            allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
            content_type = file.content_type or ""
            if content_type and content_type not in allowed_types:
                return jsonify({"error": "invalid_file_type", "message": f"不支持 {content_type} 格式，仅接受 JPG/PNG/WEBP"}), 400

            # Validate file size
            if len(image_bytes) > MAX_IMAGE_SIZE * 2:
                return jsonify({"error": "image_too_large", "message": f"图片过大 ({len(image_bytes)} bytes)，请压缩后重新上传"}), 400

            image_base64, media_type = compress_image(image_bytes)
            source_type = request.form.get("sourceType", "other_image")
            current_date = request.form.get("currentDate", current_date)
            timezone = request.form.get("timezone", timezone)
            provider = request.form.get("provider", None)

            source_hint = f"这是一张{'钉钉' if source_type == 'dingtalk_image' else '微信' if source_type == 'wechat_image' else ''}聊天截图。"
            logger.info(f"classify-image: {len(image_bytes)} bytes, source={source_type}")

            result = call_ai((image_base64, media_type), is_image=True, source_hint=source_hint, provider=provider)
            result["sourceType"] = source_type
            return jsonify(result)

        # JSON body with base64 image
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            image_data = data.get("image", "")
            if not image_data:
                return jsonify({"error": "no_input", "message": "请提供 image 字段"}), 400

            # Parse data URI
            if image_data.startswith("data:"):
                header, b64_content = image_data.split(",", 1)
                media_type = header.split(":")[1].split(";")[0]
                image_base64 = b64_content
            else:
                image_base64 = image_data
                media_type = "image/jpeg"

            image_bytes = base64.b64decode(image_base64)
            if len(image_bytes) > MAX_IMAGE_SIZE * 2:
                return jsonify({"error": "image_too_large", "message": "图片过大，请压缩后重新上传"}), 400

            image_base64, media_type = compress_image(image_bytes)
            source_type = data.get("sourceType", "other_image")
            current_date = data.get("currentDate", current_date)
            timezone = data.get("timezone", timezone)
            provider = data.get("provider", None)

            source_hint = f"这是一张{'钉钉' if source_type == 'dingtalk_image' else '微信' if source_type == 'wechat_image' else ''}聊天截图。"
            logger.info(f"classify-image: {len(image_bytes)} bytes, source={source_type}")

            result = call_ai((image_base64, media_type), is_image=True, source_hint=source_hint, provider=provider)
            result["sourceType"] = source_type
            return jsonify(result)

        else:
            return jsonify({"error": "invalid_request", "message": "请使用 JSON 或 multipart/form-data 格式"}), 400

    except ValueError as e:
        logger.error(f"Config error: {e}")
        err_msg = str(e)
        if "ARK_API_KEY" in err_msg:
            code = "ARK_KEY_MISSING"
        elif "ARK_TEXT_MODEL" in err_msg:
            code = "TEXT_MODEL_MISSING"
        elif "ARK_VISION_MODEL" in err_msg:
            code = "VISION_MODEL_MISSING"
        else:
            code = "config_error"
        return jsonify({"error": code, "message": err_msg}), 500

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return jsonify({"error": "INVALID_AI_RESPONSE", "message": "AI 返回的 JSON 格式异常，请点击重试"}), 502

    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        err = str(e)
        if "authentication" in err.lower() or "401" in err:
            return jsonify({"error": "ARK_KEY_INVALID", "message": "豆包API Key无效或已失效，请检查 .env 文件中的 ARK_API_KEY"}), 401
        elif "429" in err:
            return jsonify({"error": "rate_limit", "message": "API 调用频率超限，请稍后重试"}), 429
        elif "model" in err.lower() and ("not found" in err.lower() or "不存在" in err):
            return jsonify({"error": "MODEL_NOT_FOUND", "message": "模型ID或推理接入点不存在，请检查 .env 中的模型配置"}), 404
        elif "timeout" in err.lower() or "timed out" in err.lower():
            return jsonify({"error": "ARK_TIMEOUT", "message": "豆包响应超时，请稍后重试"}), 504
        else:
            return jsonify({"error": "api_error", "message": "AI 分析失败，请点击重试", "detail": err}), 502


@app.route("/api/ai/transcribe-voice", methods=["POST"])
def transcribe_voice():
    """
    POST /api/ai/transcribe-voice
    Reserved endpoint for server-side voice transcription.
    Currently returns a clear message that the service is not configured.
    """
    return jsonify({
        "error": "voice_not_configured",
        "message": "语音模型尚未配置。当前使用浏览器端语音识别方案（方案A），"
                   "请在前端使用浏览器 Speech API 进行语音转文字，"
                   "转写完成后调用 /api/ai/classify-text 进行分类。"
                   "如需服务端语音识别（方案B），请配置火山引擎语音识别服务。"
    }), 503


@app.route("/api/ai/correction", methods=["POST"])
def save_correction_api():
    """
    POST /api/ai/correction
    Body: {"originalInput": "...", "aiDomain": "...", "correctedDomain": "...", "aiType": "...", "correctedType": "..."}
    Saves user correction for future few-shot learning.
    """
    data = request.get_json(silent=True) or {}
    required = ["aiDomain", "correctedDomain", "aiType", "correctedType"]
    for field in required:
        if field not in data:
            return jsonify({"error": "missing_field", "message": f"缺少必填字段: {field}"}), 400

    record = {
        "originalInput": data.get("originalInput", ""),
        "aiDomain": data["aiDomain"],
        "correctedDomain": data["correctedDomain"],
        "aiType": data["aiType"],
        "correctedType": data["correctedType"],
    }
    save_correction(record)
    logger.info(f"Correction saved: {data['aiDomain']}→{data['correctedDomain']}, {data['aiType']}→{data['correctedType']}")
    return jsonify({"status": "ok", "message": "纠错记录已保存"})


# Legacy endpoint (still works, redirects internally)
@app.route("/api/ai/classify", methods=["POST"])
def classify_legacy():
    """Legacy unified endpoint — redirects to classify-text or classify-image."""
    data = request.get_json(silent=True) or {}
    if "text" in data and data["text"]:
        return classify_text()
    elif "image" in data and data["image"]:
        return classify_image()
    elif "image" in request.files:
        return classify_image()
    else:
        return jsonify({"error": "no_input", "message": "请提供 text 或 image 字段"}), 400


# ========== Health Check ==========

@app.route("/api/health", methods=["GET"])
def health():
    """Health check — returns config status without exposing API Key."""
    ark_key_set = bool(ARK_API_KEY)
    text_model_set = bool(ARK_TEXT_MODEL)
    vision_model_set = bool(ARK_VISION_MODEL)

    # Discover LAN IPs so frontend can auto-connect from mobile
    lan_ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip not in lan_ips and not ip.startswith("127."):
                lan_ips.append(ip)
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "server": "online",
        "arkConfigured": ark_key_set,
        "textModelConfigured": text_model_set,
        "visionModelConfigured": vision_model_set,
        "activeProvider": ACTIVE_PROVIDER,
        "lanIps": lan_ips,
        "port": int(os.environ.get("PORT", 5200)),
        "correctionsCount": len(load_corrections(limit=0)),
        "voiceAvailable": False,
        "checkedAt": datetime.now().isoformat(),
        "message": (
            "豆包模型已配置，可正常使用" if (ark_key_set and text_model_set)
            else "豆包模型尚未配置，请在 .env 文件中填写 ARK_API_KEY 和 ARK_TEXT_MODEL"
        )
    })


# ========== CORS ==========

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# OPTIONS for all AI endpoints — each needs unique function name
def _opt_text(): return "", 204
def _opt_image(): return "", 204
def _opt_voice(): return "", 204
def _opt_correction(): return "", 204
def _opt_legacy(): return "", 204

app.route("/api/ai/classify-text", methods=["OPTIONS"])(_opt_text)
app.route("/api/ai/classify-image", methods=["OPTIONS"])(_opt_image)
app.route("/api/ai/transcribe-voice", methods=["OPTIONS"])(_opt_voice)
app.route("/api/ai/correction", methods=["OPTIONS"])(_opt_correction)
app.route("/api/ai/classify", methods=["OPTIONS"])(_opt_legacy)


# ========== Static File Serving ==========

@app.route("/")
def index():
    """Root route: serve the main app page."""
    return static_files("personal-os.html")

@app.route("/<path:filename>")
def static_files(filename):
    file_path = Path(__file__).parent / filename
    if file_path.exists() and file_path.is_file():
        ct = {
            ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
            ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".ico": "image/x-icon",
            ".ttf": "font/ttf", ".webp": "image/webp",
        }.get(Path(filename).suffix.lower(), "application/octet-stream")
        response = app.make_response(file_path.read_bytes())
        response.headers["Content-Type"] = ct
        # Prevent iOS PWA from caching stale HTML/JS/CSS
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return "", 404


# ========== Main ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5200))
    logger.info(f"🚀 丸子的工作台 Backend Server v2 starting on port {port}")
    logger.info(f"   Active Provider: {ACTIVE_PROVIDER}")
    doubao_status = "✓ configured" if (ARK_API_KEY and ARK_TEXT_MODEL) else "✗ NOT CONFIGURED — 请在 .env 中设置 ARK_API_KEY 和 ARK_TEXT_MODEL"
    deepseek_status = "✓ configured" if DEEPSEEK_API_KEY else "✗ NOT SET"
    logger.info(f"   doubao: {doubao_status}, text_model={ARK_TEXT_MODEL or '未配置'}, vision_model={ARK_VISION_MODEL or '未配置'}")
    logger.info(f"   deepseek: {deepseek_status}")
    logger.info(f"   voice: ✗ not configured (使用浏览器Speech API方案)")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
