import unittest

from ft_agent.web.app import app


class WebTests(unittest.TestCase):
    def test_app_exists(self) -> None:
        self.assertEqual(app.title, "ft-agent")


if __name__ == "__main__":
    unittest.main()
