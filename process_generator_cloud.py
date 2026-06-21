# process_generator_cloud.py
# Cloud variant — connects to https://ollama.com instead of local Ollama.
# Requires OLLAMA_API_KEY env variable, or pass api_key= to the constructor.
# All prompt logic lives in process_generator2.py; only the connection changes here.

import os
import json
import sys
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from process_generator2 import (
    ProcessGenerator2,
    _build_extract_prompt,
    _build_reflect_prompt,
    _build_order_prompt,
    _build_order_reflect_prompt,
    _build_zone_map_prompt,
    _build_gap_fill_prompt,
    _build_conflict_check_prompt,
    _build_steps_prompt,
    _build_steps_reflect_prompt,
)

OLLAMA_CLOUD_HOST = "https://ollama.com"


class ProcessGenerator2Cloud(ProcessGenerator2):
    def __init__(
        self,
        stage1_model:   str = "gpt-oss:120b",
        stage2_model:   str = "gpt-oss:120b",
        stage3_model:   str = "gpt-oss:120b",
        stage4_model:   str = "gpt-oss:120b",
        stage5_1_model: str = "gpt-oss:120b",
        stage5_2_model: str = "gpt-oss:120b",
        stage5_3_model: str = "gpt-oss:120b",
        stage6_model:   str = "gpt-oss:120b",
        stage7_model:   str = "gpt-oss:120b",
        max_reflect_rounds: int = 2,
        api_key: str = None,
    ):
        api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError(
                "Ollama API key required. "
                "Set the OLLAMA_API_KEY environment variable or pass api_key= to the constructor."
            )

        self.max_reflect_rounds = max_reflect_rounds
        self.stage_models = {
            "1":   stage1_model,
            "2":   stage2_model,
            "3":   stage3_model,
            "4":   stage4_model,
            "5.1": stage5_1_model,
            "5.2": stage5_2_model,
            "5.3": stage5_3_model,
            "6":   stage6_model,
            "7":   stage7_model,
        }

        def _llm(model):
            return ChatOllama(
                model=model,
                temperature=0.3,
                base_url=OLLAMA_CLOUD_HOST,
                client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
            )

        self.extract_chain        = _build_extract_prompt()        | _llm(stage1_model)   | StrOutputParser()
        self.reflect_chain        = _build_reflect_prompt()        | _llm(stage2_model)   | StrOutputParser()
        self.order_chain          = _build_order_prompt()          | _llm(stage3_model)   | StrOutputParser()
        self.order_reflect_chain  = _build_order_reflect_prompt()  | _llm(stage4_model)   | StrOutputParser()
        self.zone_map_chain       = _build_zone_map_prompt()       | _llm(stage5_1_model) | StrOutputParser()
        self.gap_fill_chain       = _build_gap_fill_prompt()       | _llm(stage5_2_model) | StrOutputParser()
        self.conflict_check_chain = _build_conflict_check_prompt() | _llm(stage5_3_model) | StrOutputParser()
        self.steps_chain          = _build_steps_prompt()          | _llm(stage6_model)   | StrOutputParser()
        self.steps_reflect_chain  = _build_steps_reflect_prompt()  | _llm(stage7_model)   | StrOutputParser()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_generator_cloud.py \"your painting description\"")
        sys.exit(1)

    description = sys.argv[1]
    gen = ProcessGenerator2Cloud()
    safe_name = description[:25].replace(" ", "_").replace("'", "")
    result = gen.generate(description, output_path=f"process2_{safe_name}.json")
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2))
