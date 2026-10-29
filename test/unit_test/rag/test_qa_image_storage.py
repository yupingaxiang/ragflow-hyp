"""QA 图片旁路的顺序、失败隔离和映射模型测试。

作者：hyp
时间：2026-10-29
"""

import ast
import asyncio
import base64
import contextlib
import enum
import importlib.util
import io
import json
import pathlib
import sys
import threading
import types
import unittest
from unittest import mock

import peewee
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _png(color: str) -> bytes:
    """生成带颜色的测试图片，供顺序和真实编码验证使用。

    Args:
        color: Pillow 支持的颜色名称。

    Returns:
        PNG 图片二进制。
    """
    with Image.new("RGB", (30, 30), color) as image, io.BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


async def _run_in_thread(function, *args, **kwargs):
    """模拟项目线程执行入口并保留真实异步切换。

    Args:
        function: 需要在工作线程运行的函数。
        *args: 位置参数。
        **kwargs: 关键字参数。

    Returns:
        被调用函数的返回值。
    """
    return await asyncio.to_thread(function, *args, **kwargs)


def _load_module(redis, settings):
    """隔离外部服务初始化，加载真实图片存储模块。

    Args:
        redis: 模拟的 Redis 服务。
        settings: 注入模拟 OSS 的配置对象。

    Returns:
        待测试的实际源码模块。
    """
    modules = {
        "common": types.SimpleNamespace(settings=settings),
        "common.misc_utils": types.SimpleNamespace(
            thread_pool_exec=_run_in_thread
        ),
        "rag.utils.redis_conn": types.SimpleNamespace(REDIS_CONN=redis),
    }
    path = _ROOT / "rag/utils/qa_image_storage.py"
    spec = importlib.util.spec_from_file_location("qa_images_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def _load_model(database):
    """从真实模型源码加载列表字段和映射模型，避免初始化线上数据库。

    Args:
        database: 本地 SQLite 测试数据库。

    Returns:
        使用实际字段和索引定义的图片映射模型。
    """
    source = (_ROOT / "api/db/db_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "TextFieldType",
        "LongTextField",
        "JSONField",
        "ListField",
        "ChunkImageMapping",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    base = type(
        "TestBase",
        (peewee.Model,),
        {"Meta": type("Meta", (), {"database": database})},
    )
    namespace = {
        "Enum": enum.Enum,
        "TextField": peewee.TextField,
        "CharField": peewee.CharField,
        "BigAutoField": peewee.BigAutoField,
        "DataBaseModel": base,
        "settings": types.SimpleNamespace(DATABASE_TYPE="mysql"),
        "json_dumps": json.dumps,
        "json_loads": json.loads,
    }
    selected = ast.Module(body=definitions, type_ignores=[])
    # 仅执行仓库中固定模型的 AST，不接受外部输入。
    exec(compile(selected, "actual_db_models", "exec"), namespace)  # noqa: S102
    return namespace["ChunkImageMapping"]


class QaImageStorageTest(unittest.TestCase):
    """验证实际存储函数，不启动 Redis、MySQL 或 OSS 网络连接。"""

    def setUp(self):
        """准备实际映射模型、可控缓存和模拟 OSS。"""
        self.database = peewee.SqliteDatabase(":memory:")
        self.model = _load_model(self.database)
        self.database.create_tables([self.model])
        self.cache = {}
        self.redis = mock.Mock()
        self.redis.set.side_effect = self._cache_set
        self.redis.get.side_effect = self.cache.get
        self.redis.queue_product.return_value = True
        self.oss = mock.Mock()
        self.settings = types.SimpleNamespace(STORAGE_IMPL=self.oss)
        self.module = _load_module(self.redis, self.settings)
        self.task = {
            "id": "task1",
            "doc_id": "doc1",
            "kb_id": "kb1",
            "tenant_id": "tenant1",
        }
        self.chunks = [
            {
                "content_with_weight": "问题：测试\t回答：内容",
                "__qa_images": [_png("red"), _png("blue")],
            }
        ]

    def tearDown(self):
        """关闭仅用于本测试的数据库连接。"""
        self.database.close()

    def _cache_set(self, key, value, exp):
        """记录有过期时间的暂存内容。

        Args:
            key: 缓存键。
            value: 暂存图片列表。
            exp: 过期秒数。

        Returns:
            写入成功标记。
        """
        self.assertGreater(exp, 0)
        self.cache[key] = value
        return True

    def _message(self):
        """执行实际暂存和提交函数，返回实际队列消息。

        Returns:
            包含关联 ID 及暂存键的消息。
        """
        self.module.stage_images(self.chunks, self.task)
        job = self.task["__qa_image_jobs"][0]
        indexed = [{"id": job["chunk_id"], "img_id": "kb1-merged"}]
        self.module.enqueue_images(self.task, indexed)
        return self.redis.queue_product.call_args.args[1]

    def _store(self, message):
        """使用真实模型执行实际上传和写表函数。

        Args:
            message: 后台任务消息。
        """
        models = types.SimpleNamespace(
            DB=types.SimpleNamespace(
                connection_context=contextlib.nullcontext,
                atomic=self.database.atomic,
            ),
            ChunkImageMapping=self.model,
        )
        with mock.patch.dict(sys.modules, {"api.db.db_models": models}):
            self.module._store_images(message)

    def test_staging_preserves_order_and_removes_transient_fields(self):
        """入库前移除二进制字段，队列消息不携带图片内容。"""
        original = list(self.chunks[0]["__qa_images"])
        message = self._message()
        self.assertNotIn("__qa_images", self.chunks[0])
        self.assertNotIn("__qa_image_jobs", self.task)
        payload = json.loads(self.cache[message["input_key"]])
        self.assertEqual([base64.b64decode(blob) for blob in payload], original)
        self.assertNotIn("image_paths", message)
        self.assertEqual(message["image_count"], 2)

    def test_upload_preserves_order_and_is_idempotent(self):
        """完整上传后记录有序路径，重复任务不重复上传和新增关系。"""
        message = self._message()
        objects = {}
        self.oss.put.side_effect = lambda bucket, name, binary, **kwargs: (
            objects.__setitem__((bucket, name), binary)
        )
        self.oss.get.side_effect = lambda bucket, name, **kwargs: objects.get(
            (bucket, name)
        )
        self._store(message)
        row = self.model.get()
        self.assertEqual(row.chunk_id, message["chunk_id"])
        self.assertEqual(row.merged_image_id, "kb1-merged")
        self.assertEqual(len(set(row.image_paths)), 2)
        for image_id in row.image_paths:
            self.assertRegex(image_id, r"^kb1-[0-9a-f]{16}$")
            _, _, object_name = image_id.partition("-")
            self.assertIn(("kb1", object_name), objects)
            self.assertNotEqual(object_name, message["chunk_id"])
        colors = []
        for binary in objects.values():
            with Image.open(io.BytesIO(binary)) as image:
                colors.append(image.getpixel((0, 0)))
        self.assertGreater(colors[0][0], colors[0][2])
        self.assertGreater(colors[1][2], colors[1][0])
        self.cache.clear()
        self._store(message)
        self.assertEqual(self.model.select().count(), 1)
        self.assertEqual(self.oss.put.call_count, 2)

    def test_short_object_names_are_stable_and_scoped(self):
        """短 ID 重试稳定，图片序号、内容和知识库变化时生成不同对象名。"""
        message = self._message()
        names = self.module._image_object_names(message)
        self.assertEqual(names, self.module._image_object_names(dict(message)))
        self.assertEqual(len(set(names)), message["image_count"])
        for field in ("kb_id", "doc_id", "chunk_id", "digest"):
            changed = dict(message)
            changed[field] = f"{message[field]}-changed"
            self.assertTrue(
                set(names).isdisjoint(self.module._image_object_names(changed))
            )

    def test_silent_upload_failure_does_not_write_mapping(self):
        """模拟 OSS 吞异常，实际读回失败时不能生成成功映射。"""
        message = self._message()
        self.oss.get.return_value = None
        with self.assertRaises(OSError):
            self._store(message)
        self.assertEqual(self.model.select().count(), 0)

    def test_conflicting_images_do_not_overwrite_mapping(self):
        """相同切片出现不同图片时拒绝静默覆盖。"""
        message = self._message()
        self.model.create(
            kb_id="kb1",
            doc_id="doc1",
            chunk_id=message["chunk_id"],
            merged_image_id="kb1-merged",
            image_paths=["old-path"],
        )
        with self.assertRaises(ValueError):
            self._store(message)
        self.assertEqual(self.model.get().image_paths, ["old-path"])
        self.oss.put.assert_not_called()

    def test_cache_failure_does_not_escape(self):
        """暂存失败不影响切片正文，临时字段仍被移除。"""
        self.redis.set.side_effect = RuntimeError("Redis unavailable")
        self.module.stage_images(self.chunks, self.task)
        self.assertNotIn("__qa_images", self.chunks[0])
        self.assertIn("content_with_weight", self.chunks[0])
        self.assertNotIn("__qa_image_jobs", self.task)

    def test_dry_run_has_no_external_writes(self):
        """对比执行清理字段但不写缓存或队列。"""
        self.module.stage_images(self.chunks, self.task, dry_run=True)
        self.assertNotIn("__qa_images", self.chunks[0])
        self.redis.set.assert_not_called()

    def test_missing_merged_image_is_not_enqueued(self):
        """原有合并图未成功生成时不提交映射任务。"""
        self.module.stage_images(self.chunks, self.task)
        self.module.enqueue_images(self.task, [])
        self.redis.queue_product.assert_not_called()

    def test_word_keeps_multiple_images_without_mutating_source(self):
        """Word 同段多图按顺序保留，不操作原对象生命周期。"""
        blobs = [_png("red"), _png("blue")]
        image = types.SimpleNamespace(_blobs=blobs)
        chunk = {}
        self.module.attach_word_images(chunk, image)
        self.assertEqual(chunk["__qa_images"], blobs)
        self.assertIsNot(chunk["__qa_images"], blobs)

    def test_pdf_uses_only_figures_in_current_question(self):
        """PDF 过滤文字、表格和其他问答图片，并按位置排序。"""

        def layout(kind, top):
            """构造单页版面区域。"""
            return {
                "type": kind,
                "x0": 0,
                "x1": 10,
                "top": top,
                "bottom": top + 10,
            }

        with Image.new("RGB", (60, 300), "red") as page:
            parser = types.SimpleNamespace(
                page_from=5,
                page_images=[page],
                page_layout=[
                    [
                        layout("figure", 40),
                        layout("table", 20),
                        layout("figure", 10),
                        layout("figure", 80),
                    ]
                ],
            )
            chunk = {}
            positions = [(5, 0, 10, 10, 20), (5, 0, 10, 40, 50)]
            self.module.attach_pdf_images(chunk, parser, positions)
            self.assertEqual(len(chunk["__qa_images"]), 2)
            self.assertEqual(page.size, (60, 300))

    def test_consumer_retries_without_touching_original_status(self):
        """后台失败独立重试三次，不调用原任务进度更新。"""
        stop = threading.Event()
        message = mock.Mock()
        message.get_message.return_value = {"chunk_id": "chunk1"}
        message.ack.side_effect = lambda: stop.set() or True
        self.redis.get_unacked_iterator.return_value = iter([message])
        with (
            mock.patch.object(
                self.module, "_store_images", side_effect=OSError("OSS down")
            ) as store,
            mock.patch.object(
                self.module.asyncio, "sleep", new=mock.AsyncMock()
            ),
            mock.patch.object(self.module, "_cleanup_failed_images"),
        ):
            asyncio.run(self.module.consume_images("consumer1", stop))
        self.assertEqual(store.call_count, 3)
        message.ack.assert_called_once()


if __name__ == "__main__":
    unittest.main()
