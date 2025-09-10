import os
import sys
import tempfile
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.loaders.csv import CSVLoader
from upsonic.loaders.config import CSVLoaderConfig
from upsonic.schemas.data_models import Document

class TestCSVLoaderComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = {}
        cls.create_basic_csv(cls)
        cls.create_malformed_csv(cls)
        cls.create_encoding_test_csv(cls)
        cls.create_large_csv(cls)
        cls.create_no_header_csv(cls)
        cls.create_empty_csv(   cls)
        cls.create_special_chars_csv(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_basic_csv(self):
        file_path = os.path.join(self.temp_dir, "basic_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age', 'city', 'salary', 'department'])
            writer.writerow(['John Doe', '30', 'New York', '75000', 'Engineering'])
            writer.writerow(['Jane Smith', '25', 'San Francisco', '80000', 'Marketing'])
            writer.writerow(['Bob Johnson', '35', 'Chicago', '70000', 'Sales'])
            writer.writerow(['Alice Brown', '28', 'Boston', '85000', 'Engineering'])
            writer.writerow(['Charlie Wilson', '32', 'Seattle', '90000', 'Product'])
        self.test_files['basic'] = file_path

    @staticmethod
    def create_malformed_csv(self):
        file_path = os.path.join(self.temp_dir, "malformed_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value'])
            writer.writerow(['1', 'Item 1', '100'])
            writer.writerow(['2', 'Item 2'])  # Missing column
            writer.writerow(['3', 'Item 3', '300', 'extra'])  # Extra column
            writer.writerow(['4', 'Item 4', '400'])
        self.test_files['malformed'] = file_path

    @staticmethod
    def create_encoding_test_csv(self):
        file_path = os.path.join(self.temp_dir, "encoding_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'description'])
            writer.writerow(['José García', 'Café con leche'])
            writer.writerow(['François Müller', 'Cœur de la ville'])
            writer.writerow(['李小明', '中文测试'])
            writer.writerow(['Александр', 'Русский текст'])
        self.test_files['encoding'] = file_path

    @staticmethod
    def create_large_csv(self):
        file_path = os.path.join(self.temp_dir, "large_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'category', 'price', 'stock'])
            for i in range(1000):
                writer.writerow([
                    str(i),
                    f'Product {i}',
                    f'Category {i % 10}',
                    str(10.99 + (i % 100)),
                    str(100 - (i % 50))
                ])
        self.test_files['large'] = file_path

    @staticmethod
    def create_no_header_csv(self):
        file_path = os.path.join(self.temp_dir, "no_header_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Data 1', 'Data 2', 'Data 3'])
            writer.writerow(['Value 1', 'Value 2', 'Value 3'])
            writer.writerow(['Test 1', 'Test 2', 'Test 3'])
        self.test_files['no_header'] = file_path

    @staticmethod
    def create_empty_csv(self):
        file_path = os.path.join(self.temp_dir, "empty_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            pass  # Empty file
        self.test_files['empty'] = file_path

    @staticmethod
    def create_special_chars_csv(self):
        file_path = os.path.join(self.temp_dir, "special_chars_test.csv")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'number', 'empty', 'quotes', 'commas'])
            writer.writerow(['Normal text', '123', '', '"Quoted text"', 'Text, with, commas'])
            writer.writerow(['', '0', '', '', ''])
            writer.writerow(['   ', '  456  ', '   ', '   "   "   ', '   ,   ,   '])
        self.test_files['special_chars'] = file_path

    def test_basic_loading(self):
        loader = CSVLoader()
        documents = loader.load(self.test_files['basic'])
        self.assertEqual(len(documents), 5)
        first_doc = documents[0]
        self.assertIn('John Doe', first_doc.content)
        self.assertEqual(first_doc.metadata['name'], 'John Doe')
        self.assertEqual(first_doc.metadata['age'], '30')
        self.assertEqual(first_doc.metadata['row_number'], 1)

    def test_content_synthesis_modes(self):
        config_concatenated = CSVLoaderConfig(content_synthesis_mode="concatenated")
        loader_concatenated = CSVLoader(config_concatenated)
        docs_concatenated = loader_concatenated.load(self.test_files['basic'])
        first_doc_concatenated = docs_concatenated[0]
        self.assertIn(', ', first_doc_concatenated.content)
        self.assertIn('name: John Doe', first_doc_concatenated.content)

        config_json = CSVLoaderConfig(content_synthesis_mode="json")
        loader_json = CSVLoader(config_json)
        docs_json = loader_json.load(self.test_files['basic'])
        first_doc_json = docs_json[0]
        content_data = json.loads(first_doc_json.content)
        self.assertEqual(content_data['name'], 'John Doe')

    def test_column_filtering(self):
        config_include = CSVLoaderConfig(include_columns=['name', 'age'])
        loader_include = CSVLoader(config_include)
        docs_include = loader_include.load(self.test_files['basic'])
        first_doc_include = docs_include[0]
        self.assertIn('name: John Doe', first_doc_include.content)
        self.assertIn('age: 30', first_doc_include.content)
        self.assertNotIn('city:', first_doc_include.content)

        config_exclude = CSVLoaderConfig(exclude_columns=['salary', 'department'])
        loader_exclude = CSVLoader(config_exclude)
        docs_exclude = loader_exclude.load(self.test_files['basic'])
        first_doc_exclude = docs_exclude[0]
        self.assertNotIn('salary:', first_doc_exclude.content)
        self.assertNotIn('department:', first_doc_exclude.content)
        self.assertIn('name: John Doe', first_doc_exclude.content)

    def test_error_handling(self):
        config_warn = CSVLoaderConfig(error_handling="warn")
        loader_warn = CSVLoader(config_warn)
        docs_warn = loader_warn.load(self.test_files['malformed'])
        self.assertGreaterEqual(len(docs_warn), 2)

        config_ignore = CSVLoaderConfig(error_handling="ignore")
        loader_ignore = CSVLoader(config_ignore)
        docs_ignore = loader_ignore.load(self.test_files['malformed'])
        self.assertGreaterEqual(len(docs_ignore), 2)

    def test_no_header_csv(self):
        config_no_header = CSVLoaderConfig(has_header=False)
        loader_no_header = CSVLoader(config_no_header)
        docs_no_header = loader_no_header.load(self.test_files['no_header'])
        self.assertEqual(len(docs_no_header), 3)
        first_doc = docs_no_header[0]
        self.assertIn('column_0: Data 1', first_doc.content)
        self.assertEqual(first_doc.metadata['column_0'], 'Data 1')

    def test_empty_content_handling(self):
        config_skip = CSVLoaderConfig(skip_empty_content=True)
        loader_skip = CSVLoader(config_skip)
        docs_skip = loader_skip.load(self.test_files['special_chars'])
        self.assertLess(len(docs_skip), 4)

        config_keep = CSVLoaderConfig(skip_empty_content=False)
        loader_keep = CSVLoader(config_keep)
        docs_keep = loader_keep.load(self.test_files['special_chars'])
        self.assertGreaterEqual(len(docs_keep), 2)

    def test_custom_metadata(self):
        custom_metadata = {"source_type": "test", "version": "1.0"}
        config_custom = CSVLoaderConfig(custom_metadata=custom_metadata)
        loader_custom = CSVLoader(config_custom)
        docs_custom = loader_custom.load(self.test_files['basic'])
        first_doc = docs_custom[0]
        self.assertEqual(first_doc.metadata['source_type'], 'test')
        self.assertEqual(first_doc.metadata['version'], '1.0')

    def test_encoding_detection(self):
        loader = CSVLoader()
        docs = loader.load(self.test_files['encoding'])
        self.assertEqual(len(docs), 4)
        for doc in docs:
            self.assertTrue(doc.content)
            if 'José' in doc.content:
                self.assertIn('José', doc.content)
            elif 'François' in doc.content:
                self.assertIn('François', doc.content)

    def test_performance(self):
        loader = CSVLoader()
        start_time = time.time()
        docs = loader.load(self.test_files['large'])
        end_time = time.time()
        processing_time = end_time - start_time
        self.assertEqual(len(docs), 1000)
        self.assertLess(processing_time, 5.0)
        stats = loader.get_stats()
        self.assertEqual(stats['total_files_processed'], 1)
        self.assertEqual(stats['total_documents_created'], 1000)

    def test_file_not_found(self):
        config_ignore = CSVLoaderConfig(error_handling="ignore")
        loader_ignore = CSVLoader(config_ignore)
        docs = loader_ignore.load("non_existent_file.csv")
        self.assertEqual(len(docs), 0)

        config_warn = CSVLoaderConfig(error_handling="warn")
        loader_warn = CSVLoader(config_warn)
        docs_warn = loader_warn.load("non_existent_file.csv")
        self.assertEqual(len(docs_warn), 0)

        config_raise = CSVLoaderConfig(error_handling="raise")
        loader_raise = CSVLoader(config_raise)
        try:
            docs_raise = loader_raise.load("non_existent_file.csv")
            self.assertEqual(len(docs_raise), 0)
        except Exception as e:
            self.assertTrue("File not found" in str(e) or "not a valid file" in str(e))

    def test_batch_loading(self):
        loader = CSVLoader()
        sources = [self.test_files['basic'], self.test_files['encoding']]
        results = loader.load_batch(sources)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.success for result in results))
        self.assertEqual(sum(len(result.documents) for result in results), 9)

    def test_directory_loading(self):
        loader = CSVLoader()
        results = loader.load_directory(self.temp_dir, file_patterns=["*.csv"])
        self.assertGreaterEqual(len(results), 7)
        total_docs = sum(len(result.documents) for result in results if result.success)
        self.assertGreater(total_docs, 0)

if __name__ == "__main__":
    unittest.main()
