from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum


class ModelCapability(str, Enum):
    """Categories of model capabilities."""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    MATHEMATICS = "mathematics"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    MULTILINGUAL = "multilingual"
    VISION = "vision"
    AUDIO = "audio"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"
    COST_EFFECTIVE = "cost_effective"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"
    ETHICAL_SAFETY = "ethical_safety"
    RESEARCH = "research"
    PRODUCTION = "production"


class ModelTier(str, Enum):
    """Model performance tiers."""
    FLAGSHIP = "flagship"  # Top-tier, most capable models
    ADVANCED = "advanced"  # High performance, balanced cost
    STANDARD = "standard"  # Good performance, cost-effective
    FAST = "fast"  # Optimized for speed and low cost
    SPECIALIZED = "specialized"  # Domain-specific optimizations


@dataclass
class BenchmarkScores:
    """Performance metrics from standard AI benchmarks."""
    
    # General knowledge and reasoning
    mmlu: Optional[float] = None  # Massive Multitask Language Understanding (0-100)
    gpqa: Optional[float] = None  # Graduate-level questions (0-100)
    
    # Mathematics and problem solving
    math: Optional[float] = None  # MATH benchmark (0-100)
    gsm8k: Optional[float] = None  # Grade school math (0-100)
    aime: Optional[float] = None  # American Invitational Mathematics Examination (0-100)
    
    # Coding capabilities
    humaneval: Optional[float] = None  # Python code generation (0-100)
    mbpp: Optional[float] = None  # Mostly Basic Python Problems (0-100)
    
    # Reading comprehension
    drop: Optional[float] = None  # Discrete Reasoning Over Paragraphs (0-100)
    
    # Multilingual
    mgsm: Optional[float] = None  # Multilingual Grade School Math (0-100)
    
    # Reasoning
    arc_challenge: Optional[float] = None  # AI2 Reasoning Challenge (0-100)
    
    def overall_score(self) -> float:
        """Calculate a weighted overall score."""
        scores = []
        weights = []
        
        if self.mmlu is not None:
            scores.append(self.mmlu)
            weights.append(2.0)  # Higher weight for MMLU
        
        if self.humaneval is not None:
            scores.append(self.humaneval)
            weights.append(1.5)
        
        if self.math is not None:
            scores.append(self.math)
            weights.append(1.5)
        
        if self.gpqa is not None:
            scores.append(self.gpqa)
            weights.append(1.0)
        
        if self.gsm8k is not None:
            scores.append(self.gsm8k)
            weights.append(1.0)
        
        if self.drop is not None:
            scores.append(self.drop)
            weights.append(1.0)
        
        if not scores:
            return 0.0
        
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


@dataclass
class ModelMetadata:
    """Complete metadata for an AI model."""
    
    name: str
    provider: str
    tier: ModelTier
    release_date: str
    
    # Capabilities
    capabilities: List[ModelCapability] = field(default_factory=list)
    
    # Context window (in tokens)
    context_window: int = 8192
    
    # Performance benchmarks
    benchmarks: Optional[BenchmarkScores] = None
    
    # Strengths and ideal use cases
    strengths: List[str] = field(default_factory=list)
    ideal_for: List[str] = field(default_factory=list)
    
    # Limitations
    limitations: List[str] = field(default_factory=list)
    
    # Cost indicators (relative scale: 1-10, where 1 is cheapest)
    cost_tier: int = 5
    
    # Speed indicators (relative scale: 1-10, where 10 is fastest)
    speed_tier: int = 5
    
    # Additional notes
    notes: str = ""


# OpenAI Models
GPT_4O = ModelMetadata(
    name="openai/gpt-4o",
    provider="openai",
    tier=ModelTier.FLAGSHIP,
    release_date="2024-05-13",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.ANALYSIS,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.AUDIO,
        ModelCapability.LONG_CONTEXT,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.STRUCTURED_OUTPUT,
        ModelCapability.PRODUCTION,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=88.7,
        gpqa=53.6,
        math=76.6,
        humaneval=90.2,
        gsm8k=95.8,
        mgsm=90.5,
        drop=83.4,
    ),
    strengths=[
        "Excellent general-purpose performance across all tasks",
        "Strong multimodal capabilities (text, vision, audio)",
        "High accuracy in complex reasoning",
        "Reliable function calling and structured outputs",
        "Very large context window (128K tokens)",
    ],
    ideal_for=[
        "Production applications requiring high reliability",
        "Complex multi-step reasoning tasks",
        "Multimodal applications",
        "General-purpose AI assistants",
        "Code generation and analysis",
    ],
    limitations=[
        "Higher cost compared to smaller models",
        "Slower inference than specialized fast models",
    ],
    cost_tier=7,
    speed_tier=6,
    notes="GPT-4o is OpenAI's flagship model with excellent all-around performance. The 'o' stands for 'omni', reflecting its multimodal nature.",
)

GPT_4O_MINI = ModelMetadata(
    name="openai/gpt-4o-mini",
    provider="openai",
    tier=ModelTier.FAST,
    release_date="2024-07-18",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.FAST_INFERENCE,
        ModelCapability.COST_EFFECTIVE,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.STRUCTURED_OUTPUT,
        ModelCapability.PRODUCTION,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=82.0,
        math=70.2,
        humaneval=87.2,
        gsm8k=91.8,
        mgsm=86.7,
        drop=80.1,
    ),
    strengths=[
        "Excellent cost-performance ratio",
        "Fast inference speed",
        "Strong performance for the size",
        "Same context window as GPT-4o (128K)",
        "Suitable for high-volume applications",
    ],
    ideal_for=[
        "Cost-sensitive applications",
        "High-throughput workloads",
        "Real-time applications",
        "Chatbots and customer service",
        "Simple to moderate complexity tasks",
    ],
    limitations=[
        "Lower performance on very complex reasoning",
        "Less capable than full GPT-4o on difficult tasks",
    ],
    cost_tier=2,
    speed_tier=9,
    notes="GPT-4o-mini offers the best price-performance ratio in OpenAI's lineup. Ideal for applications where cost and speed matter.",
)

O1_PRO = ModelMetadata(
    name="openai/o1-pro",
    provider="openai",
    tier=ModelTier.SPECIALIZED,
    release_date="2025-03-19",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.MATHEMATICS,
        ModelCapability.CODE_GENERATION,
        ModelCapability.ANALYSIS,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=91.8,
        gpqa=78.3,
        math=94.8,
        aime=79.2,
        humaneval=92.5,
    ),
    strengths=[
        "Exceptional reasoning capabilities",
        "State-of-the-art performance on complex problems",
        "Extended 'thinking time' for difficult tasks",
        "Superior mathematical reasoning",
        "Excellent for research-level questions",
    ],
    ideal_for=[
        "Complex mathematical proofs",
        "Advanced coding challenges",
        "Research and analysis tasks",
        "Problems requiring deep reasoning",
        "Scientific computation",
    ],
    limitations=[
        "Slower inference due to extended reasoning",
        "Higher cost per request",
        "Overkill for simple tasks",
    ],
    cost_tier=10,
    speed_tier=3,
    notes="O1-Pro uses extended reasoning chains to solve complex problems. Best for tasks where correctness matters more than speed.",
)

O1_MINI = ModelMetadata(
    name="openai/o1-mini",
    provider="openai",
    tier=ModelTier.SPECIALIZED,
    release_date="2024-09-12",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.COST_EFFECTIVE,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=85.2,
        math=87.2,
        humaneval=89.3,
        gpqa=60.0,
    ),
    strengths=[
        "Strong reasoning at lower cost",
        "Excellent for STEM tasks",
        "Fast reasoning model",
        "Good math and coding performance",
    ],
    ideal_for=[
        "STEM applications",
        "Coding tasks requiring reasoning",
        "Mathematical problem solving",
        "Educational applications",
    ],
    limitations=[
        "Less capable than o1-pro on very hard problems",
        "No vision or audio capabilities",
    ],
    cost_tier=6,
    speed_tier=5,
    notes="O1-mini provides reasoning capabilities at a more accessible price point.",
)

# Anthropic Models
CLAUDE_4_OPUS = ModelMetadata(
    name="anthropic/claude-4-opus-20250514",
    provider="anthropic",
    tier=ModelTier.FLAGSHIP,
    release_date="2025-05-14",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.ANALYSIS,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.LONG_CONTEXT,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.ETHICAL_SAFETY,
        ModelCapability.PRODUCTION,
    ],
    context_window=200000,
    benchmarks=BenchmarkScores(
        mmlu=90.7,
        gpqa=59.4,
        math=80.5,
        humaneval=92.0,
        gsm8k=96.4,
        drop=85.3,
    ),
    strengths=[
        "Top-tier performance across all benchmarks",
        "Exceptional long context handling (200K tokens)",
        "Strong ethical guardrails and safety",
        "Excellent at following complex instructions",
        "Superior creative writing capabilities",
    ],
    ideal_for=[
        "Complex analysis of long documents",
        "Creative content generation",
        "Applications requiring safety and ethics",
        "Regulated industries (healthcare, legal, finance)",
        "High-stakes decision support",
    ],
    limitations=[
        "Higher cost",
        "Slower than smaller models",
    ],
    cost_tier=9,
    speed_tier=5,
    notes="Claude 4 Opus is Anthropic's most capable model with industry-leading context window and safety features.",
)

CLAUDE_3_7_SONNET = ModelMetadata(
    name="anthropic/claude-3-7-sonnet-20250219",
    provider="anthropic",
    tier=ModelTier.ADVANCED,
    release_date="2025-02-19",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.ANALYSIS,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.LONG_CONTEXT,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.ETHICAL_SAFETY,
        ModelCapability.PRODUCTION,
    ],
    context_window=200000,
    benchmarks=BenchmarkScores(
        mmlu=88.3,
        gpqa=54.6,
        math=78.6,
        humaneval=90.0,
        gsm8k=94.6,
        drop=84.4,
    ),
    strengths=[
        "Excellent balance of performance and cost",
        "Very large context window (200K tokens)",
        "Strong coding capabilities",
        "Good at following instructions precisely",
        "Enhanced agentic capabilities",
    ],
    ideal_for=[
        "Production AI agents",
        "Code generation and review",
        "Document analysis and summarization",
        "Multi-step workflows",
        "Complex reasoning tasks",
    ],
    limitations=[
        "Slightly lower performance than Opus tier",
    ],
    cost_tier=6,
    speed_tier=7,
    notes="Claude 3.7 Sonnet provides excellent performance for most production use cases at a reasonable cost.",
)

CLAUDE_3_5_HAIKU = ModelMetadata(
    name="anthropic/claude-3-5-haiku-20241022",
    provider="anthropic",
    tier=ModelTier.FAST,
    release_date="2024-10-22",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.FAST_INFERENCE,
        ModelCapability.COST_EFFECTIVE,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.PRODUCTION,
    ],
    context_window=200000,
    benchmarks=BenchmarkScores(
        mmlu=81.0,
        math=65.5,
        humaneval=82.0,
        gsm8k=88.3,
    ),
    strengths=[
        "Very fast inference",
        "Cost-effective",
        "Large context window (200K tokens)",
        "Good performance for the price",
        "Low latency",
    ],
    ideal_for=[
        "High-throughput applications",
        "Real-time chat applications",
        "Cost-sensitive deployments",
        "Simple to moderate tasks",
        "Customer support",
    ],
    limitations=[
        "Lower performance on complex tasks",
        "Less capable reasoning than Sonnet/Opus",
    ],
    cost_tier=2,
    speed_tier=9,
    notes="Claude 3.5 Haiku is optimized for speed and cost while maintaining good quality.",
)

# Google Models
GEMINI_2_5_PRO = ModelMetadata(
    name="google-gla/gemini-2.5-pro",
    provider="google",
    tier=ModelTier.FLAGSHIP,
    release_date="2025-06-17",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.ANALYSIS,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.AUDIO,
        ModelCapability.LONG_CONTEXT,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.PRODUCTION,
    ],
    context_window=1000000,
    benchmarks=BenchmarkScores(
        mmlu=89.5,
        gpqa=56.1,
        math=76.2,
        humaneval=88.9,
        gsm8k=94.6,
        mgsm=91.7,
        drop=84.9,
    ),
    strengths=[
        "Enormous context window (1M tokens)",
        "Excellent multimodal capabilities",
        "Strong multilingual performance",
        "Fast inference for its capability level",
        "Good integration with Google services",
    ],
    ideal_for=[
        "Processing very long documents",
        "Multimodal applications",
        "Applications integrated with Google ecosystem",
        "Multilingual applications",
        "Video and audio understanding",
    ],
    limitations=[
        "Requires Google Cloud setup",
        "Performance varies by region",
    ],
    cost_tier=7,
    speed_tier=7,
    notes="Gemini 2.5 Pro offers an unprecedented 1M token context window, ideal for processing massive amounts of data.",
)

GEMINI_2_5_FLASH = ModelMetadata(
    name="google-gla/gemini-2.5-flash",
    provider="google",
    tier=ModelTier.FAST,
    release_date="2025-06-17",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.VISION,
        ModelCapability.FAST_INFERENCE,
        ModelCapability.COST_EFFECTIVE,
        ModelCapability.LONG_CONTEXT,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.PRODUCTION,
    ],
    context_window=1000000,
    benchmarks=BenchmarkScores(
        mmlu=83.7,
        math=69.5,
        humaneval=84.7,
        gsm8k=89.7,
    ),
    strengths=[
        "Very fast inference",
        "Large context window (1M tokens)",
        "Cost-effective",
        "Good multimodal support",
        "Low latency",
    ],
    ideal_for=[
        "Real-time applications",
        "High-volume workloads",
        "Cost-sensitive deployments",
        "Simple to moderate tasks",
        "Fast document processing",
    ],
    limitations=[
        "Lower performance on complex reasoning",
    ],
    cost_tier=2,
    speed_tier=10,
    notes="Gemini 2.5 Flash combines speed and cost-effectiveness with a massive context window.",
)

# Meta Llama Models
LLAMA_3_3_70B = ModelMetadata(
    name="groq/llama-3.3-70b-versatile",
    provider="meta",
    tier=ModelTier.ADVANCED,
    release_date="2024-12-06",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.RESEARCH,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=86.0,
        math=66.0,
        humaneval=79.5,
        gsm8k=90.2,
    ),
    strengths=[
        "Open-source model",
        "Strong general performance",
        "Good code generation",
        "No vendor lock-in",
        "Can be self-hosted",
    ],
    ideal_for=[
        "Open-source projects",
        "Self-hosted deployments",
        "Research applications",
        "Custom fine-tuning",
        "Privacy-sensitive applications",
    ],
    limitations=[
        "Requires infrastructure for self-hosting",
        "Lower performance than top proprietary models",
    ],
    cost_tier=3,
    speed_tier=7,
    notes="Llama 3.3 70B is Meta's latest open-source model with strong performance across tasks.",
)

# DeepSeek Models
DEEPSEEK_R1 = ModelMetadata(
    name="deepseek/deepseek-reasoner",
    provider="deepseek",
    tier=ModelTier.SPECIALIZED,
    release_date="2025-01-20",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.MATHEMATICS,
        ModelCapability.CODE_GENERATION,
        ModelCapability.ANALYSIS,
        ModelCapability.RESEARCH,
    ],
    context_window=64000,
    benchmarks=BenchmarkScores(
        mmlu=90.8,
        math=97.3,
        aime=79.8,
        humaneval=90.2,
        gpqa=71.5,
    ),
    strengths=[
        "Exceptional reasoning capabilities",
        "State-of-the-art math performance",
        "Excellent code generation",
        "Cost-effective for capability level",
        "Strong on competitive programming",
    ],
    ideal_for=[
        "Mathematical problem solving",
        "Complex coding challenges",
        "Research tasks",
        "Competitive programming",
        "STEM education",
    ],
    limitations=[
        "Slower inference due to reasoning",
        "Less versatile than general models",
        "Primarily focused on reasoning tasks",
    ],
    cost_tier=5,
    speed_tier=4,
    notes="DeepSeek-R1 is a reasoning-focused model with exceptional math and coding performance.",
)

DEEPSEEK_CHAT = ModelMetadata(
    name="deepseek/deepseek-chat",
    provider="deepseek",
    tier=ModelTier.STANDARD,
    release_date="2024-11-04",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.COST_EFFECTIVE,
        ModelCapability.PRODUCTION,
    ],
    context_window=64000,
    benchmarks=BenchmarkScores(
        mmlu=84.5,
        math=70.8,
        humaneval=84.5,
        gsm8k=88.5,
    ),
    strengths=[
        "Very cost-effective",
        "Good general performance",
        "Strong coding capabilities",
        "Competitive pricing",
        "Reliable for production",
    ],
    ideal_for=[
        "Cost-sensitive applications",
        "General-purpose chatbots",
        "Code assistance",
        "Production deployments",
        "High-volume workloads",
    ],
    limitations=[
        "Lower performance than top-tier models",
        "Less capable on very complex tasks",
    ],
    cost_tier=1,
    speed_tier=7,
    notes="DeepSeek-Chat offers excellent value for money with solid all-around performance.",
)

# Qwen Models
QWEN_3_235B = ModelMetadata(
    name="huggingface/Qwen/Qwen3-235B-A22B",
    provider="qwen",
    tier=ModelTier.ADVANCED,
    release_date="2025-05-07",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.MULTILINGUAL,
        ModelCapability.ANALYSIS,
        ModelCapability.RESEARCH,
    ],
    context_window=32768,
    benchmarks=BenchmarkScores(
        mmlu=88.5,
        math=72.5,
        humaneval=87.2,
        gsm8k=93.4,
    ),
    strengths=[
        "Open-source large model",
        "Excellent multilingual capabilities",
        "Strong Chinese language support",
        "Good math and reasoning",
        "Active development community",
    ],
    ideal_for=[
        "Multilingual applications (especially Chinese)",
        "Open-source projects",
        "Research and development",
        "Custom fine-tuning",
        "Asian market applications",
    ],
    limitations=[
        "Large model size requires significant resources",
        "English performance slightly below top Western models",
    ],
    cost_tier=4,
    speed_tier=5,
    notes="Qwen 3 is Alibaba's flagship open-source model with exceptional multilingual capabilities.",
)

# Mistral Models
MISTRAL_LARGE = ModelMetadata(
    name="mistral/mistral-large-latest",
    provider="mistral",
    tier=ModelTier.ADVANCED,
    release_date="2024-07-24",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.PRODUCTION,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=84.0,
        math=68.0,
        humaneval=82.0,
        gsm8k=88.0,
    ),
    strengths=[
        "Good balance of performance and cost",
        "Strong European model",
        "Excellent function calling",
        "Good multilingual support",
        "Fast inference",
    ],
    ideal_for=[
        "European deployments",
        "Function calling applications",
        "Production chatbots",
        "Multilingual applications",
        "General-purpose tasks",
    ],
    limitations=[
        "Performance below top-tier US models",
    ],
    cost_tier=5,
    speed_tier=7,
    notes="Mistral Large is a strong European alternative with good all-around performance.",
)

MISTRAL_SMALL = ModelMetadata(
    name="mistral/mistral-small-latest",
    provider="mistral",
    tier=ModelTier.FAST,
    release_date="2024-07-24",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.MULTILINGUAL,
        ModelCapability.FAST_INFERENCE,
        ModelCapability.COST_EFFECTIVE,
        ModelCapability.PRODUCTION,
    ],
    context_window=32000,
    benchmarks=BenchmarkScores(
        mmlu=78.5,
        math=60.0,
        humaneval=76.2,
        gsm8k=83.5,
    ),
    strengths=[
        "Very cost-effective",
        "Fast inference",
        "Good for simple tasks",
        "European hosting options",
        "Low latency",
    ],
    ideal_for=[
        "Cost-sensitive applications",
        "Real-time chat",
        "Simple tasks",
        "High-volume workloads",
        "European data residency requirements",
    ],
    limitations=[
        "Lower performance on complex tasks",
    ],
    cost_tier=2,
    speed_tier=9,
    notes="Mistral Small provides fast, cost-effective inference for simpler tasks.",
)

# Cohere Models
COHERE_COMMAND_R_PLUS = ModelMetadata(
    name="cohere/command-r-plus",
    provider="cohere",
    tier=ModelTier.ADVANCED,
    release_date="2024-08-08",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.ANALYSIS,
        ModelCapability.MULTILINGUAL,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.PRODUCTION,
    ],
    context_window=128000,
    benchmarks=BenchmarkScores(
        mmlu=83.1,
        math=64.5,
        humaneval=78.0,
    ),
    strengths=[
        "Excellent retrieval-augmented generation (RAG)",
        "Strong enterprise features",
        "Good multilingual support",
        "Robust function calling",
        "Enterprise-ready",
    ],
    ideal_for=[
        "Enterprise RAG applications",
        "Search and retrieval",
        "Document Q&A systems",
        "Business intelligence",
        "Production enterprise apps",
    ],
    limitations=[
        "Not as strong in pure reasoning",
        "More expensive than some alternatives",
    ],
    cost_tier=6,
    speed_tier=6,
    notes="Command R+ excels at retrieval and enterprise applications.",
)

# Grok Models
GROK_4 = ModelMetadata(
    name="grok/grok-4",
    provider="grok",
    tier=ModelTier.FLAGSHIP,
    release_date="2025-07-09",
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.CODE_GENERATION,
        ModelCapability.MATHEMATICS,
        ModelCapability.CREATIVE_WRITING,
        ModelCapability.ANALYSIS,
        ModelCapability.VISION,
        ModelCapability.LONG_CONTEXT,
        ModelCapability.PRODUCTION,
    ],
    context_window=131072,
    benchmarks=BenchmarkScores(
        mmlu=89.7,
        math=81.0,
        humaneval=91.5,
        gsm8k=95.5,
    ),
    strengths=[
        "Strong overall performance",
        "Excellent reasoning",
        "Good code generation",
        "Access to real-time information (X platform)",
        "Creative and engaging responses",
    ],
    ideal_for=[
        "Applications needing current information",
        "Complex reasoning tasks",
        "Creative writing",
        "Research and analysis",
        "Social media integration",
    ],
    limitations=[
        "Requires X platform API access",
        "Limited availability",
    ],
    cost_tier=8,
    speed_tier=6,
    notes="Grok 4 is xAI's flagship model with unique access to X platform data.",
)


# Create a comprehensive registry
MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    # OpenAI
    "openai/gpt-4o": GPT_4O,
    "openai/gpt-4o-2024-11-20": GPT_4O,
    "gpt-4o": GPT_4O,
    "openai/gpt-4o-mini": GPT_4O_MINI,
    "gpt-4o-mini": GPT_4O_MINI,
    "openai/o1-pro": O1_PRO,
    "o1-pro": O1_PRO,
    "openai/o1-mini": O1_MINI,
    "o1-mini": O1_MINI,

    # Azure (using OpenAI metadata)
    "azure/gpt-4o": GPT_4O,
    "azure/gpt-4o-mini": GPT_4O_MINI,
    
    # Anthropic
    "anthropic/claude-4-opus-20250514": CLAUDE_4_OPUS,
    "claude-4-opus-20250514": CLAUDE_4_OPUS,
    "anthropic/claude-3-7-sonnet-20250219": CLAUDE_3_7_SONNET,
    "claude-3-7-sonnet-20250219": CLAUDE_3_7_SONNET,
    "anthropic/claude-3-5-haiku-20241022": CLAUDE_3_5_HAIKU,
    "claude-3-5-haiku-20241022": CLAUDE_3_5_HAIKU,
    
    # Google
    "google-gla/gemini-2.5-pro": GEMINI_2_5_PRO,
    "google-vertex/gemini-2.5-pro": GEMINI_2_5_PRO,
    "google-gla/gemini-2.5-flash": GEMINI_2_5_FLASH,
    "google-vertex/gemini-2.5-flash": GEMINI_2_5_FLASH,
    
    # Meta Llama
    "groq/llama-3.3-70b-versatile": LLAMA_3_3_70B,
    "huggingface/meta-llama/Llama-3.3-70B-Instruct": LLAMA_3_3_70B,
    
    # DeepSeek
    "deepseek/deepseek-reasoner": DEEPSEEK_R1,
    "deepseek/deepseek-chat": DEEPSEEK_CHAT,
    
    # Qwen
    "huggingface/Qwen/Qwen3-235B-A22B": QWEN_3_235B,
    "cerebras/qwen-3-235b-a22b-instruct-2507": QWEN_3_235B,
    
    # Mistral
    "mistral/mistral-large-latest": MISTRAL_LARGE,
    "mistral/mistral-small-latest": MISTRAL_SMALL,
    
    # Cohere
    "cohere/command-r-plus": COHERE_COMMAND_R_PLUS,
    "cohere/command-r-plus-08-2024": COHERE_COMMAND_R_PLUS,
    
    # Grok
    "grok/grok-4": GROK_4,
}


def get_model_metadata(model_name: str) -> Optional[ModelMetadata]:
    """
    Get metadata for a specific model.
    
    Args:
        model_name: The model name (with or without provider prefix)
    
    Returns:
        ModelMetadata if found, None otherwise
    """
    return MODEL_REGISTRY.get(model_name)


def get_models_by_capability(capability: ModelCapability) -> List[ModelMetadata]:
    """
    Get all models that have a specific capability.
    
    Args:
        capability: The capability to filter by
    
    Returns:
        List of ModelMetadata objects with the capability
    """
    return [
        model for model in MODEL_REGISTRY.values()
        if capability in model.capabilities
    ]


def get_models_by_tier(tier: ModelTier) -> List[ModelMetadata]:
    """
    Get all models in a specific tier.
    
    Args:
        tier: The tier to filter by
    
    Returns:
        List of ModelMetadata objects in the tier
    """
    return [
        model for model in MODEL_REGISTRY.values()
        if model.tier == tier
    ]


def get_top_models(n: int = 10, by_benchmark: Optional[str] = None) -> List[ModelMetadata]:
    """
    Get the top N models by overall score or specific benchmark.
    
    Args:
        n: Number of top models to return
        by_benchmark: Specific benchmark to sort by (e.g., 'mmlu', 'humaneval')
    
    Returns:
        List of top ModelMetadata objects
    """
    models_with_scores = []
    
    for model in MODEL_REGISTRY.values():
        if model.benchmarks is None:
            continue
        
        if by_benchmark:
            score = getattr(model.benchmarks, by_benchmark, None)
            if score is not None:
                models_with_scores.append((model, score))
        else:
            score = model.benchmarks.overall_score()
            if score > 0:
                models_with_scores.append((model, score))
    
    # Sort by score descending
    models_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [model for model, _ in models_with_scores[:n]]

