import unittest

from gge_assistant.rag_context import query_aspects


class CrossSourceQuestionTests(unittest.TestCase):
    def test_comparison_between_materials_keeps_both_parts(self):
        question = "В каком из материалов говорится об ориентации и масштабе сканов, а в каком — о преимуществах РИМ?"
        queries = query_aspects(question)
        self.assertEqual(len(queries), 3)
        self.assertIn("ориентации и масштабе", queries[1])
        self.assertIn("преимуществах РИМ", queries[2])


if __name__ == "__main__":
    unittest.main()
