"""QA 独立图片的暂存、异步 OSS 上传及映射持久化。

作者：hyp
时间：2026-10-29
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import threading
from typing import Any

import xxhash
from PIL import Image

from common import settings
from common.misc_utils import thread_pool_exec
from rag.utils.redis_conn import REDIS_CONN

_LOGGER = logging.getLogger(__name__)

# 使用现有 Redis 暂存跨进程任务输入；消息仅携带缓存键和关联标识。
_QUEUE = "ragflow:qa-images"
_GROUP = "qa-image-storage"
_TTL = 86400
_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
_IMAGES_FIELD = "__qa_images"
_JOBS_FIELD = "__qa_image_jobs"


def attach_word_images(chunk: dict[str, Any], image: Any) -> None:
    """将 Word 合并前的图片二进制附到临时切片字段，不改变合并图。

    Args:
        chunk: 当前问答切片。
        image: 原有 LazyImage 对象，其二进制列表按原文排列。
    """
    try:
        blobs = list(getattr(image, "_blobs", []))
        if blobs:
            chunk[_IMAGES_FIELD] = blobs
    except Exception:
        _LOGGER.exception("QA independent Word image collection failed")


def attach_pdf_images(
    chunk: dict[str, Any],
    parser: Any,
    positions: list[tuple[int, float, float, float, float]] | None,
    zoom: int = 3,
) -> None:
    """保存当前问答区域内的独立插图，保持公共 PDF 裁切逻辑不变。

    Args:
        chunk: 当前问答切片。
        parser: 保留页面截图及版面识别结果的 PDF 解析器。
        positions: 原有问答截图的有效区域坐标。
        zoom: 页面截图相对版面坐标的缩放比例。
    """
    try:
        images = []
        # 使用已有版面结果，只提取 figure，不将文字或表格截图作为图片。
        for page_index, layouts in enumerate(parser.page_layout):
            page_number = page_index + parser.page_from
            ordered_layouts = sorted(
                layouts, key=lambda item: (item["top"], item["x0"])
            )
            for layout in ordered_layouts:
                if layout.get("type") != "figure":
                    continue
                left, right = layout["x0"], layout["x1"]
                top, bottom = layout["top"], layout["bottom"]
                area = (right - left) * (bottom - top)
                if area <= 0:
                    continue
                # figure 需基本落在该问答已有区域内，避免归入相邻问答。
                belongs = any(
                    pn == page_number
                    and (
                        max(0, min(right, r) - max(left, l))
                        * max(0, min(bottom, b) - max(top, t))
                        >= area * 0.8
                    )
                    for pn, l, r, t, b in (positions or [])
                )
                if belongs:
                    bounds = (
                        left * zoom,
                        top * zoom,
                        right * zoom,
                        bottom * zoom,
                    )
                    with (
                        parser.page_images[page_index].crop(bounds) as image,
                        io.BytesIO() as stream,
                    ):
                        image.save(stream, format="PNG")
                        images.append(stream.getvalue())
        if images:
            chunk[_IMAGES_FIELD] = images
    except Exception:
        _LOGGER.exception("QA independent PDF image collection failed")


def stage_images(
    chunks: list[dict[str, Any]],
    task: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """移除临时字段并暂存图片，暂存失败不影响原有切片。

    Args:
        chunks: 尚未写入检索存储的解析结果。
        task: 当前解析任务，暂存任务描述仅保留在运行时。
        dry_run: 对比执行时仅移除临时字段，不产生外部写入。
    """
    jobs = []
    for chunk in chunks:
        blobs = chunk.pop(_IMAGES_FIELD, None)
        if not blobs or dry_run:
            continue
        try:
            # 与现有切片算法一致，不能影响现有 ID 或把图片写进检索索引。
            content = chunk["content_with_weight"] + str(task["doc_id"])
            chunk_id = xxhash.xxh64(
                content.encode("utf-8", "surrogatepass")
            ).hexdigest()
            # 在编码前限制大小，避免旁路产生不受控的临时内存占用。
            encoded_size = sum(4 * ((len(blob) + 2) // 3) + 4 for blob in blobs)
            if encoded_size > _MAX_PAYLOAD_BYTES:
                raise ValueError("QA image group exceeds temporary size limit")
            payload = json.dumps(
                [base64.b64encode(blob).decode("ascii") for blob in blobs]
            )
            digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
            key = f"{_QUEUE}:input:{task['id']}:{chunk_id}:{digest}"
            # Redis 暂存有过期时间；即使原解析失败，输入也不会永久遗留。
            if not REDIS_CONN.set(key, payload, exp=_TTL):
                raise RuntimeError("Cannot stage QA images in Redis")
            jobs.append(
                {
                    "input_key": key,
                    "chunk_id": chunk_id,
                    "digest": digest,
                    "image_count": len(blobs),
                }
            )
        except Exception:
            _LOGGER.exception(
                "QA image staging failed for document %s", task.get("doc_id")
            )
    if jobs:
        task[_JOBS_FIELD] = jobs


def enqueue_images(task: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    """切片成功入库后提交后台任务；提交失败不改变原解析状态。

    Args:
        task: 保存了暂存任务描述的解析任务。
        chunks: 已成功写入检索存储的切片，包含原有合并图标识。
    """
    try:
        indexed = {chunk["id"]: chunk for chunk in chunks}
        for job in task.pop(_JOBS_FIELD, []):
            chunk = indexed.get(job["chunk_id"], {})
            if not chunk.get("img_id"):
                continue
            message = dict(
                job,
                kb_id=task["kb_id"],
                doc_id=task["doc_id"],
                tenant_id=task["tenant_id"],
                merged_image_id=chunk["img_id"],
            )
            # 复用 Redis Stream，独立队列不占用原有解析任务消费入口。
            if not REDIS_CONN.queue_product(_QUEUE, message):
                _LOGGER.error(
                    "QA image enqueue failed for chunk %s", job["chunk_id"]
                )
    except Exception:
        _LOGGER.exception(
            "QA image enqueue failed for document %s", task.get("doc_id")
        )


def _image_object_names(message: dict[str, Any]) -> list[str]:
    """生成与原图片格式一致、任务重试时稳定的短对象名。

    Args:
        message: 包含知识库、文档、切片、图片组摘要和数量的任务。

    Returns:
        按原文顺序排列的 16 位十六进制对象名，无目录及扩展名。
    """
    # 独立命名域区分合并图，序号区分同组图片，内容摘要区分不同输入。
    seed = (
        f"qa-image:{message['kb_id']}:{message['doc_id']}:"
        f"{message['chunk_id']}:{message['digest']}"
    )
    return [
        xxhash.xxh64(f"{seed}:{index}".encode()).hexdigest()
        for index in range(message["image_count"])
    ]


def _store_images(message: dict[str, Any]) -> None:
    """上传一组图片并保存短图片 ID 列表，失败交由独立消费者重试。

    Args:
        message: 包含暂存位置、切片标识和合并图标识的队列消息。

    Raises:
        ValueError: 输入过期、图片数量错误或存在冲突映射。
        OSError: 图片解码或 OSS 上传验证失败。
        Exception: MySQL 或存储调用失败，由后台消费者统一重试。
    """
    from api.db.db_models import DB, ChunkImageMapping

    identity = {key: message[key] for key in ("kb_id", "doc_id", "chunk_id")}
    # 上传、幂等校验及失败清理共用命名规则，不能仅缩短数据库值。
    object_names = _image_object_names(message)
    expected_paths = [f"{message['kb_id']}-{name}" for name in object_names]
    with DB.connection_context():
        existing = ChunkImageMapping.get_or_none(**identity)
        if existing:
            # 仅保证同一任务重投幂等，不实现重新解析覆盖。
            if (
                existing.merged_image_id != message["merged_image_id"]
                or existing.image_paths != expected_paths
            ):
                raise ValueError("Conflicting QA image mapping")
            return

    payload = REDIS_CONN.get(message["input_key"])
    if not payload:
        raise ValueError("QA image temporary input is missing or expired")
    blobs = json.loads(payload)
    if len(blobs) != message["image_count"]:
        raise ValueError("QA image count does not match staged input")
    paths = []
    for index, encoded in enumerate(blobs):
        # 与原有合并图一样，以短 ID 为对象名，图片仍编码为 JPEG。
        name = object_names[index]
        with (
            Image.open(io.BytesIO(base64.b64decode(encoded))) as original,
            original.convert("RGB") as image,
            io.BytesIO() as stream,
        ):
            image.save(stream, format="JPEG")
            binary = stream.getvalue()
        settings.STORAGE_IMPL.put(
            message["kb_id"], name, binary, tenant_id=message["tenant_id"]
        )
        # OSS put 当前会吞异常，不能仅凭返回值判定上传成功。
        actual = settings.STORAGE_IMPL.get(
            message["kb_id"], name, tenant_id=message["tenant_id"]
        )
        if actual != binary:
            raise OSError(f"QA image upload verification failed: {name}")
        # 字段名保持 image_paths，每项与原 image_id 一样是逻辑桶加短 ID。
        paths.append(f"{message['kb_id']}-{name}")

    with DB.connection_context(), DB.atomic():
        # 同一幂等任务可能重投；冲突检查防止重复问答静默覆盖。
        row, created = ChunkImageMapping.get_or_create(
            **identity,
            defaults={
                "merged_image_id": message["merged_image_id"],
                "image_paths": paths,
            },
        )
        if not created and (
            row.image_paths != paths
            or row.merged_image_id != message["merged_image_id"]
        ):
            raise ValueError("Conflicting QA image mapping")


def _finish_message(message: Any) -> None:
    """确认并移除已处理的独立图片队列消息。

    Args:
        message: 当前独立图片队列的消息。
    """
    # 只删除本队列已确认条目，不操作原有解析任务队列。
    if message.ack():
        REDIS_CONN.REDIS.xdel(_QUEUE, message.get_msg_id())


def _cleanup_failed_images(message: dict[str, Any]) -> None:
    """重试耗尽后尽力清理未被关系表引用的独立图片。

    Args:
        message: 已耗尽重试次数的后台任务。
    """
    from api.db.db_models import DB, ChunkImageMapping

    try:
        # 无法确认数据库状态时保留对象，防止删除已提交映射的资源。
        identity = {
            key: message[key] for key in ("kb_id", "doc_id", "chunk_id")
        }
        with DB.connection_context():
            if ChunkImageMapping.get_or_none(**identity):
                return
        for name in _image_object_names(message):
            # 仅清理该独立图片任务的对象，原有合并图对象始终保留。
            settings.STORAGE_IMPL.rm(
                message["kb_id"],
                name,
                tenant_id=message["tenant_id"],
            )
        REDIS_CONN.delete(message["input_key"])
    except Exception:
        _LOGGER.exception("QA failed image cleanup could not be completed")


async def consume_images(
    consumer_name: str, stop_event: threading.Event
) -> None:
    """独立消费图片任务，不占用原有解析任务并发额度。

    Args:
        consumer_name: 原执行器名称，重启后用于恢复自己的未确认消息。
        stop_event: 执行器关闭信号。
    """
    pending = None
    while not stop_event.is_set():
        try:
            if pending is None:
                pending = iter(
                    REDIS_CONN.get_unacked_iterator(
                        [_QUEUE], _GROUP, consumer_name
                    )
                )
            message = await thread_pool_exec(next, pending, None)
            if message is None:
                message = await thread_pool_exec(
                    REDIS_CONN.queue_consumer, _QUEUE, _GROUP, consumer_name
                )
            if message is None:
                await asyncio.sleep(1)
                continue
            payload = message.get_message()
            for attempt in range(3):
                try:
                    # 所有 OSS/MySQL 阻塞调用在工作线程执行，失败不触及原任务状态。
                    await thread_pool_exec(_store_images, payload)
                    await thread_pool_exec(
                        REDIS_CONN.delete, payload["input_key"]
                    )
                    await thread_pool_exec(_finish_message, message)
                    break
                except Exception:
                    _LOGGER.exception(
                        "QA image task failed (attempt %d/3)", attempt + 1
                    )
                    if attempt < 2:
                        await asyncio.sleep(attempt + 1)
            else:
                _LOGGER.error(
                    "QA image task exhausted retries: chunk %s",
                    payload.get("chunk_id"),
                )
                await thread_pool_exec(_cleanup_failed_images, payload)
                await thread_pool_exec(_finish_message, message)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception(
                "QA image consumer failed; original parsing continues"
            )
            await asyncio.sleep(3)
