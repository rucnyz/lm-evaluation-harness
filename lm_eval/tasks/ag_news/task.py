# -*- coding: utf-8 -*-
# @Time    : 12/20/2023 9:02 PM
# @Author  : yuzhn
# @File    : task.py
# @Software: PyCharm

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from lm_eval.api.task import Task
from lm_eval.api.metrics import mean


@register_task("agnews")
class AGNEWS(Task):
    VERSION = 0
    DATASET_PATH = "ag_news"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return ("{}\nIn what section of the newspaper would you expect to find this article?"
                "\nWorld, Sports, Business, Sci/Tech\nAnswer:").format(doc["text"])

    def doc_to_target(self, doc):
        return " {}".format({0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}[doc["label"]])

    def construct_requests(self, doc, ctx, **kwargs):
        return [
            Instance(
                request_type = "loglikelihood",
                doc = doc,
                arguments = (ctx, " " + "World"),
                idx = 0,
                **kwargs
            ),
            Instance(
                request_type = "loglikelihood",
                doc = doc,
                arguments = (ctx, " " + "Sports"),
                idx = 0,
                **kwargs
            ),
            Instance(
                request_type = "loglikelihood",
                doc = doc,
                arguments = (ctx, " " + "Business"),
                idx = 0,
                **kwargs
            ),
            Instance(
                request_type = "loglikelihood",
                doc = doc,
                arguments = (ctx, " " + "Sci/Tech"),
                idx = 0,
                **kwargs
            ),
        ]

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax([i[0] for i in results])
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
