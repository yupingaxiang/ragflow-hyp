"""验证扩展召回接口的图片缺失处理、隔离范围及原响应保留。

作者：hyp
时间：2026-10-29
"""

import ast
import asyncio
import contextlib
import copy
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import peewee
import quart

from test.unit_test.rag import test_qa_image_storage as storage_tests

_ROOT = pathlib.Path(__file__).resolve().parents[4]


class RetrievalImageMappingTest(unittest.TestCase):
    """使用实际模型和 SQLite 验证批量查表与缺失映射处理。"""

    def setUp(self):
        """加载实际模型及补图函数，隔离线上数据库初始化。"""
        self.database = peewee.SqliteDatabase(":memory:")
        self.model = storage_tests._load_model(self.database)
        self.database.create_tables([self.model])
        models = types.SimpleNamespace(
            DB=types.SimpleNamespace(connection_context=contextlib.nullcontext),
            ChunkImageMapping=self.model,
        )
        path = _ROOT / "api/utils/retrieval_image_utils.py"
        spec = importlib.util.spec_from_file_location("image_utils_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"api.db.db_models": models}):
            spec.loader.exec_module(self.module)
        self.chunk = {
            "id": "chunk1",
            "dataset_id": "kb1",
            "document_id": "doc1",
            "image_id": "merged1",
            "content": "original answer",
            "similarity": 0.91,
        }

    def tearDown(self):
        """关闭本地测试数据库。"""
        self.database.close()

    def test_order_and_original_fields_are_preserved(self):
        """只补充有序图片列表，原有字段保持一致。"""
        self.model.create(
            kb_id="kb1",
            doc_id="doc1",
            chunk_id="chunk1",
            merged_image_id="merged1",
            image_paths=["image2", "image1"],
        )
        original = copy.deepcopy(self.chunk)
        self.module.enrich_retrieval_images([self.chunk])
        self.assertEqual(self.chunk.pop("image_paths"), ["image2", "image1"])
        self.assertEqual(self.chunk, original)

    def test_missing_merge_id_is_normal_and_does_not_query(self):
        """合并图为空是正常结果，不查询数据库、不记录错误。"""
        self.chunk["image_id"] = ""
        with (
            mock.patch.object(self.model, "select") as query,
            self.assertNoLogs(self.module._LOGGER),
        ):
            self.module.enrich_retrieval_images([self.chunk])
        self.assertEqual(self.chunk["image_paths"], [])
        query.assert_not_called()

    def test_unknown_merge_id_is_normal(self):
        """合并图没有对应映射时返回空列表，不记录错误。"""
        with self.assertNoLogs(self.module._LOGGER):
            self.module.enrich_retrieval_images([self.chunk])
        self.assertEqual(self.chunk["image_paths"], [])

    def test_empty_mapping_is_normal(self):
        """已存在映射的图片列表为空时保持正常响应。"""
        self.model.create(
            kb_id="kb1",
            doc_id="doc1",
            chunk_id="chunk1",
            merged_image_id="merged1",
            image_paths=[],
        )
        with self.assertNoLogs(self.module._LOGGER):
            self.module.enrich_retrieval_images([self.chunk])
        self.assertEqual(self.chunk["image_paths"], [])

    def test_mapping_is_scoped_to_dataset_and_document(self):
        """相同合并图标识不能读取其他知识库或文档的图片。"""
        for kb_id, doc_id in (("other-kb", "doc1"), ("kb1", "other-doc")):
            self.model.create(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_id="chunk1",
                merged_image_id="merged1",
                image_paths=["private-image"],
            )
        self.module.enrich_retrieval_images([self.chunk])
        self.assertEqual(self.chunk["image_paths"], [])

    def test_multiple_chunks_use_one_query(self):
        """多个切片使用一次批量查询，并保持各自列表独立。"""
        self.model.create(
            kb_id="kb1",
            doc_id="doc1",
            chunk_id="chunk1",
            merged_image_id="merged1",
            image_paths=["image1"],
        )
        chunks = [copy.deepcopy(self.chunk) for _ in range(3)]
        with mock.patch.object(
            self.database, "execute_sql", wraps=self.database.execute_sql
        ) as execute:
            self.module.enrich_retrieval_images(chunks)
        self.assertEqual(execute.call_count, 1)
        chunks[0]["image_paths"].append("local-only")
        self.assertEqual(chunks[1]["image_paths"], ["image1"])

    def test_database_error_keeps_original_retrieval(self):
        """数据库真正异常时记录错误，仍保留原召回字段。"""
        original = copy.deepcopy(self.chunk)
        with (
            mock.patch.object(
                self.model,
                "select",
                side_effect=peewee.OperationalError("down"),
            ),
            self.assertLogs(self.module._LOGGER, level="ERROR"),
        ):
            self.module.enrich_retrieval_images([self.chunk])
        self.assertEqual(self.chunk.pop("image_paths"), [])
        self.assertEqual(self.chunk, original)


class RetrievalImageRouteTest(unittest.IsolatedAsyncioTestCase):
    """执行实际新增路由，验证其仅扩展成功响应。"""

    async def asyncSetUp(self):
        """加载真实新增函数，并用可控原入口验证委托调用。"""
        self.app = quart.Quart(__name__)
        self.request_payload = None
        self.original_payload = {
            "code": 0,
            "data": {
                "total": 1,
                "doc_aggs": [{"doc_id": "doc1", "count": 1}],
                "chunks": [{"id": "chunk1", "image_id": "merged1"}],
            },
        }
        self.status = 200
        self.retrieval = mock.AsyncMock(side_effect=self._original_retrieval)
        self.enrich = mock.Mock(side_effect=self._enrich)
        source = (_ROOT / "api/apps/restful_apis/chunk_api.py").read_text(
            encoding="utf-8"
        )
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "retrieval_with_images"
        )
        namespace = {
            "manager": self.app,
            "Response": quart.Response,
            "current_app": quart.current_app,
            "RetCode": types.SimpleNamespace(SUCCESS=0),
            "retrieval_test": self.retrieval,
            "thread_pool_exec": asyncio.to_thread,
            "enrich_retrieval_images": self.enrich,
        }
        # 仅加载仓库中固定路由函数，不执行外部输入。
        exec(  # noqa: S102
            compile(
                ast.Module(body=[function], type_ignores=[]),
                "actual_retrieval_route",
                "exec",
            ),
            namespace,
        )

    async def _original_retrieval(self):
        """模拟原入口对请求和响应状态、响应头的处理。

        Returns:
            包含原有状态及自定义响应头的真实 Quart 响应。
        """
        self.request_payload = await quart.request.get_json()
        response = quart.jsonify(self.original_payload)
        response.status_code = self.status
        response.headers["X-Original-Header"] = "retained"
        return response

    def _enrich(self, chunks):
        """模拟补图操作。

        Args:
            chunks: 新响应中的切片列表。
        """
        for chunk in chunks:
            chunk["image_paths"] = ["image1", "image2"]

    async def test_success_keeps_request_and_response_contract(self):
        """原样传递请求，保留聚合、状态和响应头，只增加图片字段。"""
        request = {
            "dataset_ids": ["kb1"],
            "question": "question",
            "page": 2,
            "page_size": 3,
            "rerank_id": "rerank1",
            "use_kg": True,
            "toc_enhance": True,
            "highlight": True,
            "reference_metadata": {"include": True},
        }
        original = copy.deepcopy(self.original_payload)
        client = self.app.test_client()
        response = await client.post("/retrieval/images", json=request)
        payload = await response.get_json()
        self.assertEqual(self.request_payload, request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Original-Header"], "retained")
        self.assertEqual(
            payload["data"]["chunks"][0].pop("image_paths"),
            ["image1", "image2"],
        )
        self.assertEqual(payload, original)
        self.assertEqual(self.original_payload, original)
        self.retrieval.assert_awaited_once_with()

    async def test_original_error_is_returned_unchanged(self):
        """原入口报错时不查图片，保留错误内容和 HTTP 状态。"""
        self.original_payload = {"code": 401, "message": "Unauthorized"}
        self.status = 401
        response = await self.app.test_client().post(
            "/retrieval/images", json={}
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(await response.get_json(), self.original_payload)
        self.enrich.assert_not_called()

    async def test_authentication_exception_is_not_swallowed(self):
        """原入口的鉴权异常照常交给框架，不查询图片。"""
        from werkzeug.exceptions import Unauthorized

        self.retrieval.side_effect = Unauthorized()
        response = await self.app.test_client().post(
            "/retrieval/images", json={}
        )
        self.assertEqual(response.status_code, 401)
        self.enrich.assert_not_called()


if __name__ == "__main__":
    unittest.main()
