import unittest

from scripts import local_generalist_runtime as runtime


class LocalGeneralistRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = runtime.LocalGeneralistRuntime(seed=7, top_k=3, max_memory_turns=4)

    def test_calculator_tool_returns_expected_value(self) -> None:
        result = self.agent.respond("Calculate (12 + 8) * 3")
        self.assertEqual(result["intent"], "tool_calculator")
        self.assertEqual(result["answer"], "60")
        self.assertEqual(result["tool_name"], "calculator")

    def test_memory_store_and_recall_round_trip(self) -> None:
        first = self.agent.respond("Remember that my preferred sync time is 14:00 UTC.")
        second = self.agent.respond("What did I ask you to remember?")
        self.assertIn("Remembered", first["answer"])
        self.assertIn("14:00", second["answer"])
        self.assertIn("UTC", second["answer"])

    def test_exact_response_instruction(self) -> None:
        result = self.agent.respond("Respond exactly with: ACK READY")
        self.assertEqual(result["intent"], "response_exact")
        self.assertEqual(result["answer"], "ACK READY")

    def test_retrieval_lookup_returns_hits(self) -> None:
        result = self.agent.respond("Lookup in the corpus what mentions secure aggregation.")
        self.assertEqual(result["intent"], "retrieval_lookup")
        self.assertTrue(result["retrieval_hits"])
        self.assertIn("aggregate", result["answer"].lower())

    def test_expression_parsing_prefers_numeric_candidate(self) -> None:
        expr = runtime.sanitize_expression("What is 7 * 9")
        self.assertEqual(expr, "7 * 9")
        self.assertEqual(runtime.normalize_number(runtime.safe_eval_expression(expr)), "63")

    def test_invalid_top_k_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be > 0"):
            self.agent.respond("lookup tfidf", top_k=0)
