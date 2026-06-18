import unittest

from fastapi import HTTPException

from ft_agent.web.app import _chunk_text, _resolve_report_path


class WebTests(unittest.TestCase):
    def test_chunk_text_splits_final_answer_for_streaming(self) -> None:
        chunks = _chunk_text("abcdef.ghi", size=3)

        self.assertEqual(chunks, ["abc", "def", ".", "ghi"])

    def test_report_path_rejects_workspace_escape(self) -> None:
        with self.assertRaises(HTTPException):
            _resolve_report_path("../secret.md")


if __name__ == "__main__":
    unittest.main()
