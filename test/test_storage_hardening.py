from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from services.image_task_service import ImageTaskService
from services.log_service import LogService, _collect_urls


OWNER = {"id": "owner-1", "name": "Owner", "role": "admin"}


def make_log_service(path: Path, **overrides) -> LogService:
    defaults = dict(
        max_line_bytes=4096,
        detail_string_limit=100,
        rotate_threshold_bytes=8192,
        rotate_keep_bytes=2048,
        rotate_check_interval_secs=0.0,
        read_block_bytes=32,
    )
    defaults.update(overrides)
    return LogService(path, **defaults)


class LogServiceHardeningTests(unittest.TestCase):
    def test_add_caps_oversized_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = make_log_service(Path(tmp_dir) / "logs.jsonl")
            service.add("call", "调用完成", detail={
                "url": "data:image/png;base64," + "A" * 500_000,
                "note": "x" * 10_000,
            })
            line = service.path.read_text(encoding="utf-8").splitlines()[0]
            self.assertLessEqual(len(line.encode("utf-8")), service.max_line_bytes)
            item = json.loads(line)
            self.assertTrue(item["detail_truncated"])
            self.assertLessEqual(len(item["detail"]["url"]), service.detail_string_limit + 1)

    def test_add_replaces_detail_when_truncation_still_too_large(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = make_log_service(
                Path(tmp_dir) / "logs.jsonl",
                max_line_bytes=512,
                detail_string_limit=100,
            )
            # 400 个键 × 每个约 100 字符，截断后仍远超 512 字节上限
            detail = {f"key_{index}": "v" * 100 for index in range(400)}
            service.add("call", "调用完成", detail=detail)
            line = service.path.read_text(encoding="utf-8").splitlines()[0]
            self.assertLessEqual(len(line.encode("utf-8")), 512)
            item = json.loads(line)
            self.assertEqual(item["detail"], {"truncated": True, "reason": "detail exceeds log line cap"})

    def test_collect_urls_skips_data_uris(self) -> None:
        payload = {
            "data": [
                {"url": "data:image/png;base64,AAAA"},
                {"url": "https://example.test/a.png"},
            ],
            "urls": ["data:image/jpeg;base64,BBBB", "https://example.test/b.png"],
        }
        self.assertEqual(_collect_urls(payload), ["https://example.test/a.png", "https://example.test/b.png"])

    def test_list_reads_newest_first_with_filters_across_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = make_log_service(Path(tmp_dir) / "logs.jsonl")
            for index in range(20):
                service.add("account", f"更新账号-{index}", detail={"index": index})
            for index in range(5):
                service.add("call", f"调用完成-{index}", detail={"index": index})

            recent = service.list(limit=10)
            self.assertEqual(len(recent), 10)
            self.assertEqual(recent[0]["summary"], "调用完成-4")
            summaries = [item["summary"] for item in recent]
            self.assertEqual(summaries, [f"调用完成-{index}" for index in range(4, -1, -1)] + [f"更新账号-{index}" for index in range(19, 14, -1)])

            only_calls = service.list(type="call", limit=10)
            self.assertEqual([item["summary"] for item in only_calls], [f"调用完成-{index}" for index in range(4, -1, -1)])

            limited = service.list(limit=2)
            self.assertEqual(len(limited), 2)

    def test_list_legacy_id_is_content_based_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "logs.jsonl"
            legacy_line = json.dumps({"time": "2026-08-01 00:00:00", "type": "account", "summary": "旧格式"}, ensure_ascii=False)
            with path.open("w", encoding="utf-8") as file:
                file.write(legacy_line + "\n")
            service = make_log_service(path)
            first_ids = [item["id"] for item in service.list(limit=10)]
            service.add("account", "新记录")
            second_ids = [item["id"] for item in service.list(limit=10)]
            # 新记录追加后，旧格式条目的内容哈希 id 保持不变
            self.assertIn(first_ids[0], second_ids)

    def test_rotation_trims_file_to_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = make_log_service(Path(tmp_dir) / "logs.jsonl")
            for index in range(200):
                service.add("account", f"更新账号-{index}", detail={"padding": "p" * 64})
            size = service.path.stat().st_size
            self.assertLessEqual(size, service.rotate_threshold_bytes + 4096)
            lines = service.path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                json.loads(line)
            newest = service.list(limit=1)[0]
            self.assertEqual(newest["summary"], "更新账号-199")
            # 头部旧记录已被瘦身丢弃
            self.assertNotIn("更新账号-0", service.path.read_text(encoding="utf-8"))


class ImageTaskPayloadExpiryTests(unittest.TestCase):
    def make_service(self, path: Path, **overrides) -> ImageTaskService:
        defaults = dict(
            generation_handler=lambda _payload: {"data": [{"url": "data:image/png;base64," + "A" * 1000}]},
            edit_handler=lambda _payload: {"data": [{"url": "data:image/png;base64," + "B" * 1000}]},
            retention_days_getter=lambda: 30,
            payload_ttl_secs=60.0,
        )
        defaults.update(overrides)
        return ImageTaskService(path, **defaults)

    def test_payload_kept_within_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            service.submit_generation(
                OWNER, client_task_id="task-1", prompt="cat", model="gpt-image-2",
                size=None, base_url="http://local.test",
            )
            deadline = time.time() + 2.0
            item = None
            while time.time() < deadline:
                item = (service.list_tasks(OWNER, ["task-1"]).get("items") or [None])[0]
                if item and item.get("status") == "success":
                    break
                time.sleep(0.02)
            self.assertIsNotNone(item)
            self.assertIn("data", item)

    def test_payload_expired_after_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            service = self.make_service(path)
            service.submit_generation(
                OWNER, client_task_id="task-1", prompt="cat", model="gpt-image-2",
                size=None, base_url="http://local.test",
            )
            deadline = time.time() + 2.0
            while time.time() < deadline:
                item = (service.list_tasks(OWNER, ["task-1"]).get("items") or [None])[0]
                if item and item.get("status") == "success":
                    break
                time.sleep(0.02)
            self.assertIn("data", item)
            with service._lock:
                task = service._tasks["owner-1:task-1"]
                task["updated_ts"] = time.time() - 3600
                task["error"] = "E" * 10_000
            result = service.list_tasks(OWNER, ["task-1"])
            expired = (result.get("items") or [None])[0]
            self.assertNotIn("data", expired)
            self.assertTrue(expired.get("payload_expired"))
            self.assertLessEqual(len(expired.get("error") or ""), service.ERROR_TEXT_MAX_CHARS)
            # 持久化文件同样不再携带 data
            persisted = json.loads(path.read_text(encoding="utf-8"))["tasks"]
            self.assertTrue(all("data" not in task or not task.get("data") for task in persisted))

    def test_max_tasks_cap_drops_oldest_terminal_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json", max_tasks=10)
            with service._lock:
                now = time.time()
                for index in range(15):
                    # 相对当前时间生成时间戳（task-0 最旧），避免写死日期
                    # 超过保留期后整组被保留期清理删除，掩盖数量上限逻辑。
                    updated_ts = now - (15 - index) * 60
                    service._tasks[f"owner-1:task-{index}"] = {
                        "id": f"task-{index}",
                        "owner_id": "owner-1",
                        "status": "success",
                        "mode": "generate",
                        "model": "gpt-image-2",
                        "size": "",
                        "quality": "auto",
                        "verbatim_prompt": False,
                        "base_url": "",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_ts)),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_ts)),
                        "updated_ts": updated_ts,
                    }
            service.list_tasks(OWNER, [])
            self.assertLessEqual(len(service._tasks), 10)
            self.assertNotIn("owner-1:task-0", service._tasks)
            self.assertIn("owner-1:task-14", service._tasks)


if __name__ == "__main__":
    unittest.main()
