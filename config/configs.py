from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class GenAnswerConfig:
    bench_name: str = "arena-hard-v0.1"
    model_list: List[str] = field(default_factory=list)
    num_choices: int = 1
    max_tokens: int = 4096
    temperature: float = 0.0
    images_base_dir: str = ""

@dataclass
class EndpointInfo:
    api_type: str
    model_name: str
    system_prompt: Optional[str] = ""
    parallel: Optional[int] = 1
    tokenizer: Optional[str] = ""
    endpoints: Dict[str, str] = field(default_factory=dict)

@dataclass
class EndpointsConfig:
    endpoints: Dict[str, EndpointInfo] = field(default_factory=dict)