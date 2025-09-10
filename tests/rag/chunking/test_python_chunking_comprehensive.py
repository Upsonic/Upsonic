import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.python import PythonCodeChunkingStrategy, PythonCodeChunkingConfig
from upsonic.schemas.data_models import Document, Chunk

class TestPythonChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_simple_python_document(cls)
        cls.create_class_based_document(cls)
        cls.create_function_heavy_document(cls)
        cls.create_complex_nested_document(cls)
        cls.create_async_python_document(cls)
        cls.create_decorated_code_document(cls)
        cls.create_large_python_document(cls)
        cls.create_empty_python_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_simple_python_document(self):
        python_content = """import os
import sys
from typing import List, Dict

class SimpleClass:
    \"\"\"A simple class for demonstration.\"\"\"
    
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        return self.name

def simple_function(value: int) -> int:
    \"\"\"A simple function that doubles a value.\"\"\"
    return value * 2

if __name__ == "__main__":
    obj = SimpleClass("test")
    result = simple_function(5)
    print(f"Name: {obj.get_name()}, Result: {result}")"""
        doc = Document(content=python_content, metadata={'source': 'simple.py', 'language': 'python', 'type': 'basic'})
        self.test_documents['simple'] = doc

    @staticmethod
    def create_class_based_document(self):
        python_content = """class DataProcessor:
    \"\"\"A comprehensive data processing class.\"\"\"
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.processed_count = 0
    
    def preprocess_data(self) -> None:
        \"\"\"Preprocess the data before analysis.\"\"\"
        for item in self.data:
            if 'invalid' in item.get('status', ''):
                self.data.remove(item)
    
    def analyze_data(self) -> Dict[str, Any]:
        \"\"\"Analyze the preprocessed data.\"\"\"
        results = {
            'total_items': len(self.data),
            'average_value': 0,
            'max_value': 0,
            'min_value': float('inf')
        }
        
        if self.data:
            values = [item.get('value', 0) for item in self.data]
            results['average_value'] = sum(values) / len(values)
            results['max_value'] = max(values)
            results['min_value'] = min(values)
        
        return results

class AdvancedProcessor(DataProcessor):
    \"\"\"Advanced data processor with additional features.\"\"\"
    
    def __init__(self, data: List[Dict], config: Dict):
        super().__init__(data)
        self.config = config
        self.cache = {}
    
    def process_with_cache(self, key: str) -> Any:
        \"\"\"Process data with caching mechanism.\"\"\"
        if key in self.cache:
            return self.cache[key]
        
        result = self.complex_processing(key)
        self.cache[key] = result
        return result
    
    def complex_processing(self, key: str) -> Any:
        \"\"\"Complex processing logic.\"\"\"
        # Simulate complex processing
        return f"processed_{key}"""
        doc = Document(content=python_content, metadata={'source': 'classes.py', 'language': 'python', 'type': 'class_based'})
        self.test_documents['classes'] = doc

    @staticmethod
    def create_function_heavy_document(self):
        python_content = """def calculate_fibonacci(n: int) -> int:
    \"\"\"Calculate fibonacci number recursively.\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

def calculate_fibonacci_iterative(n: int) -> int:
    \"\"\"Calculate fibonacci number iteratively.\"\"\"
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def prime_generator(limit: int):
    \"\"\"Generate prime numbers up to limit.\"\"\"
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    for num in range(2, limit + 1):
        if is_prime(num):
            yield num

def factorial(n: int) -> int:
    \"\"\"Calculate factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def gcd(a: int, b: int) -> int:
    \"\"\"Calculate greatest common divisor.\"\"\"
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    \"\"\"Calculate least common multiple.\"\"\"
    return abs(a * b) // gcd(a, b)

def matrix_multiply(matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
    \"\"\"Multiply two matrices.\"\"\"
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])
    
    if cols1 != rows2:
        raise ValueError("Cannot multiply matrices with incompatible dimensions")
    
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result"""
        doc = Document(content=python_content, metadata={'source': 'functions.py', 'language': 'python', 'type': 'function_heavy', 'functions': 7})
        self.test_documents['functions'] = doc

    @staticmethod
    def create_complex_nested_document(self):
        python_content = """class ComplexSystem:
    \"\"\"A complex system with nested structures.\"\"\"
    
    class InnerProcessor:
        \"\"\"Inner class for specific processing tasks.\"\"\"
        
        def __init__(self, parent_system):
            self.parent = parent_system
            self.processed_items = []
        
        def process_item(self, item):
            \"\"\"Process a single item.\"\"\"
            def validate_item(data):
                \"\"\"Nested validation function.\"\"\"
                required_fields = ['id', 'type', 'value']
                return all(field in data for field in required_fields)
            
            if validate_item(item):
                processed = {
                    'original': item,
                    'processed_at': time.time(),
                    'processor_id': id(self)
                }
                self.processed_items.append(processed)
                return processed
            return None
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = self.InnerProcessor(self)
        self.stats = {
            'total_processed': 0,
            'errors': 0,
            'start_time': time.time()
        }
    
    def batch_process(self, items: List[Dict]) -> Dict:
        \"\"\"Process a batch of items.\"\"\"
        results = []
        
        for item in items:
            try:
                if self.should_process(item):
                    result = self.processor.process_item(item)
                    if result:
                        results.append(result)
                        self.stats['total_processed'] += 1
                    else:
                        self.handle_processing_error(item, "Validation failed")
                        self.stats['errors'] += 1
            except Exception as e:
                self.handle_processing_error(item, str(e))
                self.stats['errors'] += 1
        
        return {
            'results': results,
            'stats': self.get_processing_stats()
        }
    
    def should_process(self, item: Dict) -> bool:
        \"\"\"Determine if item should be processed.\"\"\"
        filters = self.config.get('filters', {})
        
        for filter_key, filter_value in filters.items():
            if item.get(filter_key) != filter_value:
                return False
        
        return True
    
    def handle_processing_error(self, item: Dict, error_message: str):
        \"\"\"Handle processing errors.\"\"\"
        print(f"Error processing item {item.get('id', 'unknown')}: {error_message}")
    
    def get_processing_stats(self) -> Dict:
        \"\"\"Get current processing statistics.\"\"\"
        elapsed_time = time.time() - self.stats['start_time']
        return {
            'total_processed': self.stats['total_processed'],
            'errors': self.stats['errors'],
            'elapsed_time': elapsed_time,
            'processing_rate': self.stats['total_processed'] / elapsed_time if elapsed_time > 0 else 0
        }"""
        doc = Document(content=python_content, metadata={'source': 'complex.py', 'language': 'python', 'type': 'complex_nested', 'complexity': 'high'})
        self.test_documents['complex'] = doc

    @staticmethod
    def create_async_python_document(self):
        python_content = """import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, AsyncGenerator

class AsyncDataFetcher:
    \"\"\"Asynchronous data fetching class.\"\"\"
    
    def __init__(self, base_url: str, max_concurrent: int = 10):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_single(self, endpoint: str) -> Dict:
        \"\"\"Fetch data from a single endpoint.\"\"\"
        async with self.semaphore:
            url = f"{self.base_url}/{endpoint}"
            async with self.session.get(url) as response:
                return await response.json()
    
    async def fetch_batch(self, endpoints: List[str]) -> List[Dict]:
        \"\"\"Fetch data from multiple endpoints concurrently.\"\"\"
        tasks = [self.fetch_single(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if not isinstance(result, Exception)]

async def async_file_processor(file_path: str) -> AsyncGenerator[str, None]:
    \"\"\"Process file lines asynchronously.\"\"\"
    async with aiofiles.open(file_path, 'r') as file:
        async for line in file:
            processed_line = await process_line_async(line.strip())
            if processed_line:
                yield processed_line

async def process_line_async(line: str) -> str:
    \"\"\"Process a single line asynchronously.\"\"\"
    # Simulate async processing
    await asyncio.sleep(0.001)
    return line.upper() if line else ""

async def main():
    \"\"\"Main async function.\"\"\"
    async with AsyncDataFetcher("https://api.example.com") as fetcher:
        endpoints = ["users", "posts", "comments"]
        data = await fetcher.fetch_batch(endpoints)
        
        # Process files asynchronously
        async for processed_line in async_file_processor("data.txt"):
            print(processed_line)
        
        return data

if __name__ == "__main__":
    asyncio.run(main())"""
        doc = Document(content=python_content, metadata={'source': 'async_code.py', 'language': 'python', 'type': 'async', 'paradigm': 'asynchronous'})
        self.test_documents['async'] = doc

    @staticmethod
    def create_decorated_code_document(self):
        python_content = """from functools import wraps, lru_cache
from typing import Callable, Any
import time

def timer(func: Callable) -> Callable:
    \"\"\"Decorator to measure execution time.\"\"\"
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    \"\"\"Decorator to retry function execution.\"\"\"
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class MathOperations:
    \"\"\"Class with various decorated methods.\"\"\"
    
    @staticmethod
    @lru_cache(maxsize=128)
    def fibonacci(n: int) -> int:
        \"\"\"Calculate fibonacci with caching.\"\"\"
        if n <= 1:
            return n
        return MathOperations.fibonacci(n - 1) + MathOperations.fibonacci(n - 2)
    
    @timer
    @retry(max_attempts=3, delay=0.5)
    def complex_calculation(self, data: List[int]) -> float:
        \"\"\"Perform complex calculation with timing and retry.\"\"\"
        if not data:
            raise ValueError("Data cannot be empty")
        
        result = sum(x ** 2 for x in data) / len(data)
        return result
    
    @property
    def description(self) -> str:
        \"\"\"Property with description.\"\"\"
        return "Mathematical operations class"
    
    @classmethod
    def create_default(cls):
        \"\"\"Create default instance.\"\"\"
        return cls()

@timer
def decorated_function(x: int, y: int) -> int:
    \"\"\"Simple decorated function.\"\"\"
    return x * y + x ** y

@retry(max_attempts=5)
@timer
def multiple_decorators_function(data: str) -> str:
    \"\"\"Function with multiple decorators.\"\"\"
    if not data:
        raise ValueError("Data is required")
    return data.upper().replace(" ", "_")"""
        doc = Document(content=python_content, metadata={'source': 'decorators.py', 'language': 'python', 'type': 'decorated', 'patterns': 'decorator'})
        self.test_documents['decorated'] = doc

    @staticmethod
    def create_large_python_document(self):
        lines = [
            "\"\"\"Large Python module for performance testing.\"\"\"",
            "import os",
            "import sys",
            "import time",
            "from typing import List, Dict, Any, Optional",
            ""
        ]
        
        # Create multiple classes
        for i in range(10):
            lines.extend([
                f"class DataProcessor{i}:",
                f"    \"\"\"Data processor class number {i}.\"\"\"",
                "",
                f"    def __init__(self, config: Dict):",
                f"        self.config = config",
                f"        self.data = []",
                f"        self.processed_count = 0",
                "",
            ])
            
            # Add methods to each class
            for j in range(5):
                lines.extend([
                    f"    def process_method_{j}(self, data: List[Any]) -> Dict:",
                    f"        \"\"\"Processing method {j} for class {i}.\"\"\"",
                    f"        results = {{}}",
                    f"        for item in data:",
                    f"            if isinstance(item, dict):",
                    f"                results[f'item_{{len(results)}}'] = item",
                    f"        return results",
                    "",
                ])
        
        # Create standalone functions
        for i in range(20):
            lines.extend([
                f"def utility_function_{i}(param1: Any, param2: Any = None) -> Any:",
                f"    \"\"\"Utility function number {i}.\"\"\"",
                f"    if param2 is None:",
                f"        param2 = 'default_value_{i}'",
                f"    ",
                f"    result = {{",
                f"        'function_id': {i},",
                f"        'param1': param1,",
                f"        'param2': param2,",
                f"        'timestamp': time.time()",
                f"    }}",
                f"    ",
                f"    return result",
                "",
            ])
        
        lines.extend([
            "if __name__ == '__main__':",
            "    print('Large module loaded successfully')",
            "    for i in range(10):",
            "        processor = DataProcessor0({'test': True})",
            "        result = utility_function_0('test_data')",
            "        print(f'Processed item {i}: {result}')"
        ])
        
        python_content = "\n".join(lines)
        doc = Document(content=python_content, metadata={'source': 'large_module.py', 'language': 'python', 'type': 'performance_test', 'classes': 10, 'functions': 20})
        self.test_documents['large'] = doc

    @staticmethod
    def create_empty_python_document(self):
        doc = Document(content="", metadata={'source': 'empty.py', 'language': 'python', 'type': 'edge_case'})
        self.test_documents['empty'] = doc

    def test_basic_python_chunking(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Check Python-specific content preservation
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("class SimpleClass", all_content)
        self.assertIn("def simple_function", all_content)

    def test_class_preservation(self):
        config = PythonCodeChunkingConfig(preserve_class_integrity=True)
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['classes'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify class content is preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("class DataProcessor", all_content)
        self.assertIn("class AdvancedProcessor", all_content)
        self.assertIn("def preprocess_data", all_content)

    def test_function_preservation(self):
        config = PythonCodeChunkingConfig(preserve_function_integrity=True)
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['functions'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify function content is preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("def calculate_fibonacci", all_content)
        self.assertIn("def prime_generator", all_content)
        self.assertIn("def matrix_multiply", all_content)

    def test_docstring_preservation(self):
        config = PythonCodeChunkingConfig(preserve_docstrings=True)
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify docstrings are preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("A simple class for demonstration", all_content)
        self.assertIn("A simple function that doubles", all_content)

    def test_import_context_preservation(self):
        config = PythonCodeChunkingConfig(include_imports_context=True)
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify imports are preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("import os", all_content)
        self.assertIn("from typing import", all_content)

    def test_complex_nested_structures(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify nested structures are handled
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("class ComplexSystem", all_content)
        self.assertIn("class InnerProcessor", all_content)
        self.assertIn("def validate_item", all_content)

    def test_async_code_handling(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['async'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify async code is preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("async def", all_content)
        self.assertIn("await", all_content)
        self.assertIn("AsyncGenerator", all_content)

    def test_decorator_preservation(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['decorated'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify decorators are preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("@timer", all_content)
        self.assertIn("@retry", all_content)
        self.assertIn("@lru_cache", all_content)
        self.assertIn("@property", all_content)

    def test_ast_analysis_configuration(self):
        config = PythonCodeChunkingConfig(enable_ast_analysis=True)
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['simple'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should work with AST analysis enabled
        for chunk in chunks:
            self.assertIsInstance(chunk.text_content, str)
            self.assertGreater(len(chunk.text_content), 0)

    def test_code_complexity_detection(self):
        config = PythonCodeChunkingConfig(detect_code_complexity=True)
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['complex'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify complex code structures are handled
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("for item in items", all_content)
        self.assertIn("try:", all_content)
        self.assertIn("except Exception", all_content)

    def test_empty_content_handling(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['empty'])
        
        self.assertEqual(len(chunks), 0)

    def test_whitespace_only_content(self):
        whitespace_content = "   \n\n  \t  \n"
        whitespace_doc = Document(content=whitespace_content, metadata={'type': 'whitespace'})
        
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(whitespace_doc)
        
        self.assertEqual(len(chunks), 0)

    def test_single_line_code(self):
        single_line_content = "print('Hello, World!')"
        single_line_doc = Document(content=single_line_content, metadata={'type': 'single_line'})
        
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(single_line_doc)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text_content, single_line_content)

    def test_performance_with_large_document(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 1)  # Should split large document
        self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds
        
        # Verify content preservation in large document
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("class DataProcessor", all_content)
        self.assertIn("def utility_function", all_content)

    def test_metadata_inheritance(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['decorated'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify that original document metadata is inherited
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source'], 'decorators.py')
            self.assertEqual(chunk.metadata['language'], 'python')
            self.assertEqual(chunk.metadata['type'], 'decorated')
            self.assertEqual(chunk.metadata['patterns'], 'decorator')

    def test_custom_configuration_options(self):
        # Test with various configuration combinations
        configs = [
            PythonCodeChunkingConfig(preserve_class_integrity=False),
            PythonCodeChunkingConfig(preserve_function_integrity=False),
            PythonCodeChunkingConfig(include_imports_context=False),
            PythonCodeChunkingConfig(preserve_docstrings=False),
        ]
        
        for config in configs:
            chunker = PythonCodeChunkingStrategy(config)
            chunks = chunker.chunk(self.test_documents['simple'])
            self.assertGreaterEqual(len(chunks), 0)

    def test_batch_processing(self):
        documents = [
            self.test_documents['simple'],
            self.test_documents['classes'],
            self.test_documents['functions']
        ]
        
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = PythonCodeChunkingConfig(enable_caching=True)
        chunker = PythonCodeChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['simple'])
        
        # Second processing (should use cache if available)
        chunks2 = chunker.chunk(self.test_documents['simple'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    def test_error_handling(self):
        config = PythonCodeChunkingConfig()
        chunker = PythonCodeChunkingStrategy(config)
        
        # Test with malformed Python code
        malformed_content = """def incomplete_function(
        # Missing closing parenthesis and function body
        class IncompleteClass
        # Missing colon"""
        malformed_doc = Document(content=malformed_content, metadata={'type': 'malformed'})
        
        chunks = chunker.chunk(malformed_doc)
        self.assertGreaterEqual(len(chunks), 0)  # Should handle gracefully

if __name__ == "__main__":
    unittest.main()