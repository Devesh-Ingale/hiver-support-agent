"""The agent: retrieve -> one structured LLM call -> (repair) -> deterministic policy -> run record.

`Agent.handle(item)` returns a flat dict that is written to outputs/runs/<system>.jsonl. It keeps both
what the model said (model_decision, model_reply, ...) and what the policy layer decided, so every
downstream number can be recomputed offline without another model call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from ..data.clean import clean_text, detect_lang
from ..llm.base import LLMClient, LLMError
from ..retrieval.index import Evidence, TfidfIndex
from ..taxonomy import Taxonomy
from . import prompts
from .checks import post_process
from .schemas import AgentOutput, fallback_output, output_schema

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    system: str                      # run name: main | no_rag | main_gemini ...
    use_retrieval: bool = True
    k: int = 5
    min_similarity: float = 0.0      # "no relevant resolution" floor; tuned on dev, frozen before test
    max_reply_chars: int = 280
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 700


class Agent:
    def __init__(self, llm: LLMClient, taxonomy: Taxonomy, config: AgentConfig, index: TfidfIndex | None = None):
        if config.use_retrieval and index is None:
            raise ValueError("retrieval enabled but no index given")
        self.llm = llm
        self.taxonomy = taxonomy
        self.config = config
        self.index = index
        self.hard_rules = taxonomy.hard_rules()
        self.system_prompt = prompts.render_system(taxonomy)

    # -- one message ------------------------------------------------------------------------------------
    def handle(self, item: dict) -> dict:
        message: str = item["text"]
        lang: str = item.get("lang") or detect_lang(message)
        evidence = self._retrieve(message, exclude_ids={str(item.get("doc_id", ""))})
        evidence_ids = [f"E{i}" for i in range(1, len(evidence) + 1)]
        schema = output_schema(self.taxonomy.intent_ids, evidence_ids or None)
        user_prompt = prompts.render_user(message, evidence, self.taxonomy.brand)

        response, output, parse_failed, repaired, intent_invalid = self._call_and_parse(user_prompt, schema)
        retrieval_max_sim = TfidfIndex.max_similarity(evidence) if self.config.use_retrieval else None
        result = post_process(
            output, message, lang=lang, evidence_texts=[e.brand_turns for e in evidence],
            retrieval_max_sim=retrieval_max_sim, hard_rules=self.hard_rules,
            min_similarity=self.config.min_similarity, max_chars=self.config.max_reply_chars,
        )
        final = result.output
        id_map = dict(zip(evidence_ids, (e.doc_id for e in evidence)))
        return {
            "item_id": item["item_id"],
            "system": self.config.system,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "prompt_version": prompts.PROMPT_VERSION,
            "taxonomy_version": self.taxonomy.version,
            "text": message,
            "lang": lang,
            # final (policy-applied) answer
            "intent": final.intent,
            "intent_secondary": final.intent_secondary,
            "intent_confidence": final.intent_confidence,
            "reply": final.reply,
            "decision": final.decision,
            "reason_code": final.reason_code,
            "reason": final.reason,
            "evidence_ids": [id_map.get(e, e) for e in final.evidence_ids],
            # what the model said before the policy layer
            "model_decision": result.model_decision,
            "model_reason_code": output.reason_code,
            "forced_reason": result.forced_reason,
            "violations": result.violations,
            "retrieval_max_sim": retrieval_max_sim,
            "evidence": [e.to_dict() for e in evidence],
            "parse_failed": parse_failed,
            "repaired": repaired,
            "intent_invalid": intent_invalid,
            "latency_ms": response.latency_ms if response else None,
            "cached": response.cached if response else None,
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "raw_text": response.text if response else None,
        }

    # -- helpers ---------------------------------------------------------------------------------------
    def _retrieve(self, message: str, exclude_ids: set[str]) -> list[Evidence]:
        if not self.config.use_retrieval or self.index is None:
            return []
        query = clean_text(message, brand=self.taxonomy.brand)
        return self.index.search(query, k=self.config.k, exclude_ids=exclude_ids)

    def _call_and_parse(self, user_prompt: str, schema: dict):
        """One call, one repair attempt, then a safe fallback. Never raises on bad model output."""
        response = None
        try:
            response = self._complete(user_prompt, schema)
            output = self._validate(response.json())
            return response, output, False, False, self._intent_invalid(output)
        except (ValidationError, ValueError, TypeError) as first_error:
            log.info("model output invalid (%s); attempting repair", type(first_error).__name__)
        except LLMError:
            raise
        try:
            response = self._complete(user_prompt + prompts.REPAIR_SUFFIX, schema)
            output = self._validate(response.json())
            return response, output, False, True, self._intent_invalid(output)
        except (ValidationError, ValueError, TypeError) as second_error:
            log.warning("model output unusable after repair: %s", second_error)
            return response, fallback_output(self.taxonomy.other_intent, type(second_error).__name__), True, True, False

    def _complete(self, user_prompt: str, schema: dict):
        return self.llm.complete(self.system_prompt, user_prompt, schema=schema, temperature=self.config.temperature,
                                 seed=self.config.seed, max_tokens=self.config.max_tokens)

    def _validate(self, obj) -> AgentOutput:
        if obj is None:
            raise ValueError("no JSON object in model output")
        output = AgentOutput.model_validate(obj)
        if output.intent not in self.taxonomy.intent_ids:
            output = output.model_copy(update={"intent": self.taxonomy.other_intent})
            output._intent_invalid = True  # type: ignore[attr-defined]
        if output.intent_secondary is not None and output.intent_secondary not in self.taxonomy.intent_ids:
            output = output.model_copy(update={"intent_secondary": None})
        return output

    @staticmethod
    def _intent_invalid(output: AgentOutput) -> bool:
        return bool(getattr(output, "_intent_invalid", False))
