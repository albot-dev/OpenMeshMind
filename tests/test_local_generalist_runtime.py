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

    def test_memory_list_and_forget(self) -> None:
        self.agent.respond("Remember this note: alpha checkpoint")
        self.agent.respond("Remember this note: beta checkpoint")
        listed = self.agent.respond("list my notes")
        self.assertEqual(listed["intent"], "memory_list")
        self.assertIn("alpha checkpoint", listed["answer"])
        self.assertIn("beta checkpoint", listed["answer"])

        forgot = self.agent.respond("forget the latest note")
        self.assertEqual(forgot["intent"], "memory_forget")
        self.assertIn("Forgot note", forgot["answer"])
        listed_after = self.agent.respond("show my notes")
        self.assertIn("alpha checkpoint", listed_after["answer"])
        self.assertNotIn("beta checkpoint", listed_after["answer"])

    def test_calculator_followup_uses_last_result(self) -> None:
        first = self.agent.respond("Calculate 10 + 5")
        self.assertEqual(first["answer"], "15")
        second = self.agent.respond("multiply that by 3")
        self.assertEqual(second["answer"], "45")
        self.assertEqual(second["tool_name"], "calculator")

    def test_calculator_chain_then_steps(self) -> None:
        result = self.agent.respond("Calculate 12 + 8 then multiply by 3 then subtract 5")
        self.assertEqual(result["intent"], "tool_calculator")
        self.assertEqual(result["answer"], "55")

    def test_retrieval_top_k_prompt_override(self) -> None:
        result = self.agent.respond("Lookup in the corpus what mentions secure aggregation top 2")
        self.assertEqual(result["intent"], "retrieval_lookup")
        self.assertGreaterEqual(len(result["retrieval_hits"]), 2)
        self.assertIn("1.", result["answer"])

    def test_memory_store_and_recall_with_natural_phrasing(self) -> None:
        stored = self.agent.respond("Can you remember that the demo starts Friday at 3 PM UTC?")
        recalled = self.agent.respond("Can you remind me what I asked you to remember?")
        self.assertEqual(stored["intent"], "memory_store")
        self.assertEqual(recalled["intent"], "memory_recall")
        self.assertIn("Friday", recalled["answer"])
        self.assertIn("3 PM UTC", recalled["answer"])

    def test_memory_list_with_natural_question(self) -> None:
        self.agent.respond("Remember that alpha checkpoint is on.")
        listed = self.agent.respond("What do you remember?")
        self.assertEqual(listed["intent"], "memory_list")
        self.assertIn("alpha checkpoint", listed["answer"])

    def test_retrieval_supports_word_top_k_and_docs_phrase(self) -> None:
        result = self.agent.respond(
            "What do the docs say about secure aggregation? Show three results with citations."
        )
        self.assertEqual(result["intent"], "retrieval_lookup")
        self.assertGreaterEqual(len(result["retrieval_hits"]), 3)
        self.assertIn("1.", result["answer"])

    def test_calculator_supports_word_numbers(self) -> None:
        result = self.agent.respond("What is twelve plus eight")
        self.assertEqual(result["intent"], "tool_calculator")
        self.assertEqual(result["answer"], "20")

    def test_calculator_supports_subtract_from_form(self) -> None:
        result = self.agent.respond("Subtract five from twenty")
        self.assertEqual(result["intent"], "tool_calculator")
        self.assertEqual(result["answer"], "15")

    def test_calculator_chain_supports_word_numbers(self) -> None:
        result = self.agent.respond("What is twelve plus eight then multiply by three then subtract two")
        self.assertEqual(result["intent"], "tool_calculator")
        self.assertEqual(result["answer"], "58")
