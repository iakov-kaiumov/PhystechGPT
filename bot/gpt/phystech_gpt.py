"""ChatGPT (GPT-3.5) language model from OpenAI + internal answers database"""

import re
import openai
import json
import Levenshtein as lev
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bot.config import BASE_DIR
from bot.gpt.base_gpt import BaseGPT
from bot.models import UserMessage
from bot import config

openai.api_key = config.openai_api_key

BASE_PROMPT = "Ваша главная цель - ответить на мои вопросы. " \
              "Это может включать в себя написание кода или предоставление полезной информации. " \
              "Будьте подробны и скрупулезны в своих ответах."

PRE_RE = re.compile(r"&lt;(/?pre)")


class PhystechGPT(BaseGPT):
    """OpenAI API wrapper."""
    DATABASE_DIR = BASE_DIR / 'dataset/dataset_with_keywords.json'

    def __init__(self):
        self.dataset = self._load_dataset()
        self.stop_words = set(stopwords.words('russian'))

    async def ask(self, question: str, history: list[UserMessage]) -> str:
        """Asks the language model a question and returns an answer."""
        try:
            messages = self._generate_messages(question, history)
            resp = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            answer = self._prepare_answer(resp)
            return answer

        except openai.error.InvalidRequestError as exc:
            raise ValueError("too many tokens to make completion") from exc

    def _generate_messages(self, question: str, history: list[UserMessage]) -> list[dict]:
        """Builds message history to provide context for the language model."""
        messages = [{"role": "system", "content": BASE_PROMPT}]

        matches = self._find_internal_matches(question)
        for match in matches:
            messages.append({"role": "user", "content": "При ответе, учитывай следующую информацию: \n" + match['item']['question'] + "\n" + match['item']['answer']})
            # messages.append({"role": "user", "content": match['item']['question']})
            # messages.append({"role": "assistant", "content": match['item']['answer']})
        messages.append({"role": "user", "content": question})
        print(messages)
        return messages

    def _prepare_answer(self, resp):
        """
        Post-processes an answer from the language model.
        """
        if len(resp.choices) == 0:
            raise ValueError("received an empty answer")
        answer = resp.choices[0].message.content
        return answer

    # Database methods
    def _load_dataset(self) -> list:
        with open(self.DATABASE_DIR, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_match_score(self, q: str, keywords: list[str], threshold: int = 2) -> float:
        q_tokens = word_tokenize(q.lower())
        q_words = [w for w in q_tokens if w not in self.stop_words]
        score = 0
        for word in q_words:
            for keyword in keywords:
                dist = lev.distance(word, keyword)
                if dist == 0:
                    score += 10
                elif dist == 1:
                    score += 3
                elif dist == 2:
                    score += 0.5
        return score / len(keywords)

    def _find_internal_matches(self, question: str, threshold: float = 0.5) -> list:
        matches = []
        for item in self.dataset:
            score = self._get_match_score(question, item['keywords'])
            if score > threshold:
                matches.append({'item': item, 'score': score})

        matches = sorted(matches, key=lambda x: x['score'], reverse=True)[0:5]
        return matches
