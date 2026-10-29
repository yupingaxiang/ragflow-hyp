"""为新召回接口补充 QA 独立图片，不改变原有召回结果字段。

作者：hyp
时间：2026-10-29
"""

import logging
from typing import Any

from api.db.db_models import DB, ChunkImageMapping

_LOGGER = logging.getLogger(__name__)


def enrich_retrieval_images(chunks: list[dict[str, Any]]) -> None:
    """按召回切片的知识库、文档和合并图批量补充有序图片列表。

    合并图为空、没有映射或图片列表为空均是正常情况，不记录错误。
    数据库查询异常时记录日志，保留召回字段并返回空图片列表。

    Args:
        chunks: 新接口独立解析出的响应切片，不传入共享检索结果对象。
    """
    keys = set()
    for chunk in chunks:
        chunk["image_paths"] = []
        key = (
            chunk.get("dataset_id"),
            chunk.get("document_id"),
            chunk.get("image_id"),
        )
        if all(isinstance(value, str) and value for value in key):
            keys.add(key)
    if not keys:
        return

    try:
        # 查询范围取自已鉴权的召回结果，不能只按合并图 ID 跨知识库查表。
        with DB.connection_context():
            rows = (
                ChunkImageMapping.select(
                    ChunkImageMapping.kb_id,
                    ChunkImageMapping.doc_id,
                    ChunkImageMapping.merged_image_id,
                    ChunkImageMapping.image_paths,
                )
                .where(
                    ChunkImageMapping.kb_id.in_({key[0] for key in keys}),
                    ChunkImageMapping.doc_id.in_({key[1] for key in keys}),
                    ChunkImageMapping.merged_image_id.in_(
                        {key[2] for key in keys}
                    ),
                )
                .dicts()
            )
            paths_by_key = {
                (row["kb_id"], row["doc_id"], row["merged_image_id"]): list(
                    row["image_paths"]
                )
                for row in rows
                if isinstance(row["image_paths"], list)
            }
    except Exception:
        # 补图属于旁路能力，查询失败不能覆盖已成功的召回结果。
        _LOGGER.exception("Retrieval image mapping query failed")
        return

    for chunk in chunks:
        key = (
            chunk.get("dataset_id"),
            chunk.get("document_id"),
            chunk.get("image_id"),
        )
        if all(isinstance(value, str) and value for value in key):
            chunk["image_paths"] = list(paths_by_key.get(key, []))
