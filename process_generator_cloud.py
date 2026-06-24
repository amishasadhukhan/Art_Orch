# process_generator_cloud.py
# Cloud variant — connects to https://ollama.com instead of local Ollama.
# Requires OLLAMA_API_KEY env variable, or pass api_key= to the constructor.
# All prompt and pipeline logic lives in process_generator_refined.py.
# Only the LLM connection changes here.

import os
import json
import sys
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from process_generator_refined import (
    ProcessGeneratorRefined,
    _build_s1_prompt,
    _build_s2_prompt,
    _build_s3_prompt,
    _build_s4_prompt,
    _build_s51_prompt,
    _build_s52_prompt,
    _build_s6_prompt,
    _build_s7_prompt,
    _build_s8_prompt,
    _build_s9_prompt,
    _build_s10_prompt,
)

OLLAMA_CLOUD_HOST = "https://ollama.com"


class ProcessGeneratorRefinedCloud(ProcessGeneratorRefined):
    """
    Cloud variant of ProcessGeneratorRefined.
    Connects to Ollama Cloud instead of a local Ollama instance.
    Inherits all pipeline logic (run_stage* methods) from ProcessGeneratorRefined.
    Only the LLM connection is overridden here.
    """

    def __init__(
        self,
        stage1_model:   str = "gpt-oss:120b",
        stage2_model:   str = "gpt-oss:120b",
        stage3_model:   str = "gpt-oss:120b",
        stage4_model:   str = "gpt-oss:120b",
        stage51_model:  str = "gpt-oss:120b",
        stage52_model:  str = "gpt-oss:120b",
        stage6_model:   str = "gpt-oss:120b",
        stage7_model:   str = "gpt-oss:120b",
        stage8_model:   str = "gpt-oss:120b",
        stage9_model:   str = "gpt-oss:120b",
        stage10_model:  str = "gpt-oss:120b",
        max_reflect_rounds: int = 2,
        api_key: str = None,
    ):
        api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError(
                "Ollama Cloud API key required. "
                "Set OLLAMA_API_KEY environment variable or pass api_key= to the constructor."
            )

        self.max_reflect_rounds = max_reflect_rounds
        self.stage_models = {
            "1":   stage1_model,
            "2":   stage2_model,
            "3":   stage3_model,
            "4":   stage4_model,
            "5.1": stage51_model,
            "5.2": stage52_model,
            "6":   stage6_model,
            "7":   stage7_model,
            "8":   stage8_model,
            "9":   stage9_model,
            "10":  stage10_model,
        }

        def _llm(model):
            return ChatOllama(
                model=model,
                temperature=0.3,
                base_url=OLLAMA_CLOUD_HOST,
                client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
            )

        self.extract_chain        = _build_s1_prompt()  | _llm(stage1_model)  | StrOutputParser()
        self.reflect_chain        = _build_s2_prompt()  | _llm(stage2_model)  | StrOutputParser()
        self.order_chain          = _build_s3_prompt()  | _llm(stage3_model)  | StrOutputParser()
        self.order_reflect_chain  = _build_s4_prompt()  | _llm(stage4_model)  | StrOutputParser()
        self.layout_chain         = _build_s51_prompt() | _llm(stage51_model) | StrOutputParser()
        self.scene_graph_chain    = _build_s52_prompt() | _llm(stage52_model) | StrOutputParser()
        self.layout_reflect_chain = _build_s6_prompt()  | _llm(stage6_model)  | StrOutputParser()
        self.object_realize_chain = _build_s7_prompt()  | _llm(stage7_model)  | StrOutputParser()
        self.object_reflect_chain = _build_s8_prompt()  | _llm(stage8_model)  | StrOutputParser()
        self.step_gen_chain       = _build_s9_prompt()  | _llm(stage9_model)  | StrOutputParser()
        self.step_reflect_chain   = _build_s10_prompt() | _llm(stage10_model) | StrOutputParser()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_generator_cloud.py \"description\" [output.json] [WxH]")
        sys.exit(1)

    description      = sys.argv[1]
    output_path      = sys.argv[2] if len(sys.argv) > 2 else "process_out.json"
    canvas_w, canvas_h = 900, 600
    if len(sys.argv) > 3:
        canvas_w, canvas_h = map(int, sys.argv[3].split("x"))

    gen    = ProcessGeneratorRefinedCloud()
    result = gen.run_stages_1_to_10(description, canvas_w, canvas_h, output_path)
    print("\n" + "=" * 60)
    print(json.dumps(result.get("process_json", {}), indent=2))
