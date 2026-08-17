import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.fake import FakeKVSender
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFakePrefillBackend(unittest.TestCase):
    @staticmethod
    def _make_sender() -> FakeKVSender:
        return FakeKVSender(
            mgr=MagicMock(),
            bootstrap_addr="unused:0",
            bootstrap_room=1,
            dest_tp_ranks=[0],
            pp_rank=0,
        )

    def test_prefill_fake_skips_bootstrap_server(self):
        server_args = SimpleNamespace(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="fake",
        )

        with patch(
            "sglang.srt.managers.disagg_service.get_kv_class"
        ) as mock_get_kv_class:
            self.assertIsNone(start_disagg_service(server_args))

        mock_get_kv_class.assert_not_called()

    def test_final_empty_chunk_completes_sender(self):
        sender = self._make_sender()

        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))

        sender.send(
            np.empty(0, dtype=np.int32),
            state_indices=[[np.array([7], dtype=np.int32)]],
            num_kv_tokens=0,
        )

        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_nonempty_middle_chunk_can_send(self):
        sender = self._make_sender()

        self.assertTrue(sender.should_send_kv_chunk(1, last_chunk=False))
        sender.send(np.array([3], dtype=np.int32), num_kv_tokens=1)
        self.assertEqual(sender.poll(), KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
