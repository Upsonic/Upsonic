"""
Test file to verify tool call printing with long inputs and outputs.
Runs with debug_level=2 to show full tool call table.
"""

from upsonic import Agent, Task
from upsonic.tools import tool


@tool
def analyze_document(document_content: str, analysis_type: str) -> dict:
    """
    Analyze a document and return comprehensive analysis results.
    
    Args:
        document_content: The full content of the document to analyze
        analysis_type: Type of analysis to perform (summary, keywords, sentiment)
    
    Returns:
        Comprehensive analysis results including summary, keywords, and insights
    """
    # Return a large, detailed response
    return {
        "document_length": len(document_content),
        "word_count": len(document_content.split()),
        "analysis_type": analysis_type,
        "summary": f"This document discusses multiple topics including AI, machine learning, natural language processing, and data science. The content covers theoretical foundations, practical applications, and future directions. Key themes include automation, intelligent systems, and human-AI collaboration.",
        "keywords": [
            "artificial intelligence", "machine learning", "deep learning", 
            "neural networks", "natural language processing", "data science",
            "automation", "intelligent systems", "algorithms", "optimization",
            "training data", "model architecture", "inference", "prediction"
        ],
        "key_insights": [
            "AI is transforming multiple industries including healthcare, finance, and manufacturing",
            "Machine learning models require large amounts of quality training data",
            "Natural language processing enables human-like text understanding",
            "Deep learning has achieved breakthrough results in computer vision",
            "Ethical considerations are crucial for responsible AI development"
        ],
        "sentiment": "positive and informative",
        "readability_score": 72.5,
        "topics_covered": ["AI Fundamentals", "ML Techniques", "NLP Applications", "Future Trends"],
        "recommendations": [
            "Consider adding more practical examples",
            "Include case studies from real-world implementations",
            "Add a section on ethical considerations"
        ]
    }


def main():
    print("=" * 100)
    print("TOOL CALL PRINTING TEST - Long Input/Output")
    print("=" * 100)
    
    # Create agent with debug_level=2
    agent = Agent(
        name="Document Analyzer",
        model="openai/gpt-4o-mini",
        tools=[analyze_document],
        debug=True,
        debug_level=2
    )
    
    # Long document content
    long_document = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
    From its theoretical foundations in the 1950s to today's sophisticated deep learning systems, AI has evolved 
    dramatically and is now reshaping industries, economies, and societies worldwide.
    
    Machine Learning, a subset of AI, enables systems to learn from data without being explicitly programmed. 
    This paradigm shift has led to breakthrough applications in image recognition, speech processing, and 
    natural language understanding. Deep learning, powered by neural networks with multiple layers, has 
    achieved remarkable results in complex pattern recognition tasks.
    
    Natural Language Processing (NLP) represents a particularly exciting frontier. Large language models 
    can now understand context, generate human-like text, and perform complex reasoning tasks. These 
    capabilities are being deployed in chatbots, content generation, translation services, and code assistance.
    
    The future of AI holds immense promise. Advances in reinforcement learning, multimodal AI, and 
    neuromorphic computing are opening new possibilities. However, with great power comes great responsibility - 
    ethical AI development, bias mitigation, and ensuring beneficial outcomes for humanity remain critical priorities.
    
    Key applications include autonomous vehicles, medical diagnosis, financial forecasting, climate modeling,
    drug discovery, personalized education, smart cities, and robotic automation. The economic impact is 
    projected to be in the trillions of dollars over the coming decades.
    """
    
    # Create task
    task = Task(
        description=f"Analyze this document for keywords and sentiment: {long_document}"
    )
    
    # Run agent
    result = agent.do(task)
    
    print("\n" + "=" * 100)
    print("RESULT:")
    print("=" * 100)
    print(result)


if __name__ == "__main__":
    main()
