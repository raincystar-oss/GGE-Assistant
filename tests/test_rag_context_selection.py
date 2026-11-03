"""General boundaries of the context selector; no webinar-specific fixtures."""
import tempfile
from pathlib import Path
import unittest

from gge_assistant.rag_context import ContextConfig, ContextIndex, coverage_terms, query_aspects


class ScopeTests(unittest.TestCase):
    def test_noun_list_is_not_split_into_unrelated_queries(self):
        question = 'Чем в системе АБВ отличаются ошибка, замечание и уведомление? Нужно ли исправлять каждое из них?'
        self.assertEqual(query_aspects(question), [question])

    def test_implicit_clause_inherits_only_explicit_shared_abbreviation(self):
        question = 'Как в АБВ загрузить файл? Как отправить полученный отчет?'
        queries = query_aspects(question)
        self.assertEqual(queries[0], question)
        self.assertIn('Как отправить полученный отчет (АБВ)', queries)
        self.assertLessEqual(len(queries), 3)

    def test_two_named_systems_keep_their_own_scope(self):
        question = 'Как выгрузить файл из АБВ и как распечатать протокол в ЖЗИ?'
        queries = query_aspects(question)
        self.assertNotIn('ЖЗИ', queries[1])
        self.assertNotIn('АБВ', queries[2])

    def test_dependent_why_clause_stays_in_full_question_only(self):
        question = 'В АБВ строки без связи отмечены белым цветом, и почему это важно для заказчика?'
        queries = query_aspects(question)
        self.assertEqual(queries[0], question)
        self.assertEqual(len(queries), 2)
        self.assertIn('строки без связи', queries[1])

    def test_short_independent_action_is_still_searched(self):
        question = 'Как в АБВ получить отчёт и как подписать документ?'
        self.assertTrue(any('подписать документ' in q for q in query_aspects(question)[1:]))

    def test_inflection_score_does_not_conflate_numbers_or_short_codes(self):
        self.assertEqual(coverage_terms('ошибка'), coverage_terms('ошибки'))
        self.assertEqual(coverage_terms('сметы'), coverage_terms('смета'))
        for a, b in [('115', '1150'), ('1.3', '1.30'), ('ПД', 'ПДН')]:
            self.assertFalse(coverage_terms(a) & coverage_terms(b))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.index = ContextIndex(Path(self.temp.name) / 'context.sqlite3')

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def add(self, mid, title, segments):
        self.index.upsert({'material_id': mid, 'metadata': {'title': title}, 'segments': segments},
                          {'discovery_url': 'https://edu.gge.ru/example'},
                          [{'title': mid, 'start': segments[0]['start'], 'end': segments[-1]['end']}])

    def test_context_dependent_answer_keeps_preceding_condition(self):
        self.index.cfg = ContextConfig(parent_seconds=30, parent_max_seconds=60,
                                       continuation_units=0, max_parents=3)
        self.add('a', 'Сервис АБВ', [
            {'start': 0, 'end': 35, 'text': 'Для подачи требуется доверенность. Она действует сутки.'},
            {'start': 40, 'end': 75, 'text': 'А также прикрепите подпись. Подайте обращение кнопкой регистрации.'},
        ])
        result = self.index.assemble(['Прикрепить подпись и подать обращение',
                                     'Прикрепить подпись', 'Подать обращение'], [[], [], []])
        self.assertIn('доверенность', result['context'])
        self.assertIn('действует сутки', result['context'])

    def test_distinct_core_sections_are_reserved_before_continuations(self):
        self.index.cfg = ContextConfig(child_chars=600, child_max_chars=1000,
                                       parent_seconds=30, parent_max_seconds=60,
                                       parent_chars=600, parent_max_chars=1200,
                                       context_chars=3000, continuation_units=2, max_parents=2)
        for mid, name, term in [('alpha', 'Сервис АБВ', 'Экспорт'), ('beta', 'Сервис ЖЗИ', 'Печать')]:
            segments = [
                {'start': 0, 'end': 35, 'text': 'Посторонняя длинная часть. ' * 30},
                {'start': 40, 'end': 75, 'text': (term + ' результата ' + name + '. ') * 25},
                {'start': 80, 'end': 115, 'text': 'Общие сведения, вопросы слушателей. ' * 25},
            ]
            self.add(mid, name, segments)
        result = self.index.assemble(['Экспорт АБВ и печать ЖЗИ', 'Экспорт АБВ', 'Печать ЖЗИ'],
                                     [[], [], []], [('alpha', 1), ('beta', 1)])
        self.assertEqual({b['material_id'] for b in result['blocks']}, {'alpha', 'beta'})
        self.assertIn('Экспорт результата', result['context'])
        self.assertIn('Печать результата', result['context'])
        self.assertLessEqual(result['context_chars'], 3000)

    def test_retrieval_does_not_claim_the_question_is_answerable(self):
        self.add('a', 'Инструкция', [{'start': 0, 'end': 30, 'text': 'Общие сведения об отчете.'}])
        result = self.index.assemble(['Как оплатить отчет?'], [[]])
        self.assertEqual(result['evidence_status'], 'retrieved_unverified')
        empty = self.index.assemble(['Несуществующиймаркер'], [[]])
        self.assertEqual(empty['evidence_status'], 'no_candidates')


if __name__ == '__main__':
    unittest.main()
