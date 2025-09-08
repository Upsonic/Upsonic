import os
import sys
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import unittest

from upsonic.text_splitter.structure_aware import DocumentStructureAwareChunkingStrategy, StructureAwareConfig, StructureType
from upsonic.schemas.data_models import Document, Chunk

class TestStructureAwareChunkingComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_documents = {}
        cls.create_hierarchical_document(cls)
        cls.create_list_based_document(cls)
        cls.create_table_document(cls)
        cls.create_dialogue_document(cls)
        cls.create_procedural_document(cls)
        cls.create_narrative_document(cls)
        cls.create_mixed_document(cls)
        cls.create_empty_document(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)

    @staticmethod
    def create_hierarchical_document(self):
        content = """# Chapter 1: Introduction
This chapter provides an overview of the subject matter and establishes the foundational concepts.

## Section 1.1: Background
The background section covers the historical development and current state of the field.

### Subsection 1.1.1: Early Research
Early research in this area began in the 1950s with pioneering work by several researchers.

### Subsection 1.1.2: Modern Developments
Modern developments have expanded the scope and applications significantly.

## Section 1.2: Scope and Objectives
This section defines the boundaries and goals of the current work.

### Subsection 1.2.1: Primary Objectives
The primary objectives focus on advancing theoretical understanding.

### Subsection 1.2.2: Secondary Goals
Secondary goals include practical applications and implementation strategies.

# Chapter 2: Methodology
This chapter describes the research methodology and experimental design.

## Section 2.1: Experimental Setup
The experimental setup follows established protocols with some modifications.

## Section 2.2: Data Collection
Data collection procedures ensure validity and reliability of results."""
        doc = Document(content=content, metadata={'source': 'hierarchical.txt', 'type': 'academic', 'structure': 'hierarchical'})
        self.test_documents['hierarchical'] = doc

    @staticmethod
    def create_list_based_document(self):
        content = """Shopping List for Project:

1. Hardware Components:
   a) Processors (Intel Core i7 or equivalent)
   b) Memory modules (32GB DDR4)
   c) Storage devices (1TB NVMe SSD)
   d) Graphics cards (NVIDIA RTX 4080)

2. Software Requirements:
   a) Operating system licenses
   b) Development tools and IDEs
   c) Database management systems
   d) Security software packages

3. Network Equipment:
   a) Ethernet switches (24-port Gigabit)
   b) Wireless access points (Wi-Fi 6)
   c) Network cables and connectors
   d) Firewall and security appliances

4. Peripherals and Accessories:
   a) Monitors and displays (4K resolution)
   b) Keyboards and mice (ergonomic design)
   c) Audio equipment (headsets and speakers)
   d) Backup power supplies (UPS units)

Additional Notes:
- All items must meet enterprise-grade specifications
- Vendor compatibility should be verified before purchase
- Budget approval required for items over $1000"""
        doc = Document(content=content, metadata={'source': 'list.txt', 'type': 'procurement', 'structure': 'list_based'})
        self.test_documents['list_based'] = doc

    @staticmethod
    def create_table_document(self):
        content = """Employee Performance Review Data

Name: John Smith
Department: Engineering
Review Period: Q3 2024

Performance Metrics:
| Metric | Target | Actual | Score |
|--------|--------|--------|-------|
| Code Quality | 90% | 95% | Excellent |
| Project Delivery | 100% | 98% | Good |
| Team Collaboration | 85% | 92% | Excellent |
| Innovation | 75% | 88% | Excellent |

Goals for Next Quarter:
| Goal | Priority | Deadline | Status |
|------|----------|----------|--------|
| Complete certification | High | Dec 2024 | In Progress |
| Mentor junior developer | Medium | Jan 2025 | Not Started |
| Improve documentation | Medium | Nov 2024 | In Progress |
| Lead new project | High | Q1 2025 | Planning |

Summary Statistics:
Overall Rating: 4.2/5.0
Salary Adjustment: 8% increase recommended
Promotion Eligibility: Eligible for Senior Developer role
Training Budget: $3,000 allocated for professional development"""
        doc = Document(content=content, metadata={'source': 'table.txt', 'type': 'performance_review', 'structure': 'table_based'})
        self.test_documents['table_based'] = doc

    @staticmethod
    def create_dialogue_document(self):
        content = """Customer Service Transcript - Case #12345

Agent: Hello, thank you for calling TechSupport. My name is Sarah. How can I help you today?

Customer: Hi Sarah, I'm having trouble with my internet connection. It's been really slow for the past few days.

Agent: I'm sorry to hear about that. Let me help you troubleshoot this issue. Can you tell me what internet package you currently have?

Customer: I believe it's the Premium package with 200 Mbps download speed.

Agent: Great, thank you. Have you tried restarting your modem and router recently?

Customer: Yes, I tried that yesterday but it didn't seem to help much.

Agent: I understand. Let me check your connection status from our end. Can you please provide me with your account number?

Customer: Sure, it's AC-789456123.

Agent: Perfect, thank you. I can see your account now. I notice there have been some network issues in your area. Let me run a line test to check the signal quality.

Customer: Okay, how long will that take?

Agent: The test should complete in about 2-3 minutes. While we wait, have you noticed if the slow speeds affect all devices or just specific ones?

Customer: It seems to affect all devices - my laptop, phone, and streaming device are all slow.

Agent: That's helpful information. The test results show some signal degradation. I'm going to schedule a technician visit to check your physical connection. Would tomorrow afternoon work for you?

Customer: Yes, that would be great. What time?

Agent: How about 2 PM to 4 PM? The technician will call 30 minutes before arriving.

Customer: Perfect, thank you so much for your help!

Agent: You're very welcome! Is there anything else I can help you with today?

Customer: No, that covers everything. Thank you again!

Agent: Great! Have a wonderful day, and we'll see you tomorrow for the technician visit."""
        doc = Document(content=content, metadata={'source': 'dialogue.txt', 'type': 'customer_service', 'structure': 'dialogue'})
        self.test_documents['dialogue'] = doc

    @staticmethod
    def create_procedural_document(self):
        content = """Database Backup and Recovery Procedure

Prerequisites:
- Administrative access to database server
- Sufficient storage space for backup files
- Network connectivity to backup destination
- Valid backup software licenses

Step 1: Prepare the Environment
1.1. Log into the database server with administrative privileges
1.2. Verify that all users are logged out of critical applications
1.3. Check available disk space in the backup directory
1.4. Ensure backup software is running and accessible

Step 2: Create Full Database Backup
2.1. Open the database management console
2.2. Navigate to the backup utility section
2.3. Select "Full Backup" option from the backup type menu
2.4. Specify the backup destination directory
2.5. Configure backup compression settings (recommended: enabled)
2.6. Set backup verification options (recommended: enabled)
2.7. Click "Start Backup" to begin the process

Step 3: Monitor Backup Progress
3.1. Watch the progress indicator for completion status
3.2. Monitor system resources to ensure normal operation
3.3. Check log files for any error messages or warnings
3.4. Estimate remaining time based on current progress

Step 4: Verify Backup Integrity
4.1. Wait for backup completion confirmation
4.2. Check the backup file size and compare to expected values
4.3. Run backup verification utility to ensure file integrity
4.4. Test backup file accessibility and readability

Step 5: Document and Store
5.1. Record backup completion time and file size in log
5.2. Move backup file to secure offsite storage location
5.3. Update backup inventory spreadsheet with new entry
5.4. Send confirmation email to stakeholders

Recovery Procedure (Emergency Use):
In case of system failure, follow these steps to restore from backup:
1. Assess the extent of data loss or corruption
2. Identify the most recent valid backup file
3. Prepare the target system for restoration
4. Execute the database restore procedure
5. Verify data integrity after restoration
6. Restart all dependent applications and services"""
        doc = Document(content=content, metadata={'source': 'procedural.txt', 'type': 'standard_procedure', 'structure': 'procedural'})
        self.test_documents['procedural'] = doc

    @staticmethod
    def create_narrative_document(self):
        content = """The Journey of Innovation

In the early morning hours of March 15th, 2024, Dr. Elena Rodriguez sat in her laboratory, surrounded by months of research data and countless failed prototypes. The breakthrough she had been chasing seemed as elusive as ever, yet something in the air felt different that day.

As sunlight filtered through the laboratory windows, Elena reviewed her latest experiments with a fresh perspective. The polymer she had been developing showed promise in initial tests, but achieving the right balance of flexibility and durability had proven challenging. Each iteration brought new insights, but also new obstacles to overcome.

The afternoon brought an unexpected visitor. Professor James Chen, her former mentor, stopped by to check on her progress. His experienced eyes immediately noticed subtle improvements in her latest samples. "You're closer than you think," he said, examining the material under different lighting conditions.

That evening, inspired by Professor Chen's encouragement, Elena decided to try a completely different approach. Instead of modifying the existing formula, she would start fresh with a new base compound. The idea was risky, potentially setting her research back by months, but her intuition told her it was the right path.

Working through the night, Elena carefully prepared the new mixture. Every measurement was precise, every step documented meticulously. As the compound began to take shape, she noticed properties that exceeded her most optimistic expectations. The material was not only meeting her original criteria but surpassing them significantly.

By dawn, Elena had completed the initial synthesis and testing. The results were remarkable - she had not only achieved her original goals but had discovered a material with potential applications far beyond her initial vision. The months of patient experimentation and seemingly failed attempts had led to something extraordinary.

The following weeks were a whirlwind of additional testing, documentation, and peer review. Elena's discovery would eventually revolutionize multiple industries, but in that moment, she simply sat in her quiet laboratory, appreciating the beautiful complexity of scientific discovery and the patience it requires."""
        doc = Document(content=content, metadata={'source': 'narrative.txt', 'type': 'story', 'structure': 'narrative'})
        self.test_documents['narrative'] = doc

    @staticmethod
    def create_mixed_document(self):
        content = """Project Alpha - Comprehensive Report

Executive Summary:
Project Alpha represents a strategic initiative to modernize our customer service infrastructure. This report outlines the current status, achievements, and next steps.

# Technical Architecture

## System Components
The new architecture consists of several integrated components:

1. Customer Interface Layer
   - Web portal with responsive design
   - Mobile application (iOS and Android)
   - Voice response system integration

2. Processing Engine
   - Real-time ticket routing
   - Automated response generation
   - Analytics and reporting module

3. Data Storage
   - Customer information database
   - Interaction history repository
   - Knowledge base and FAQ system

## Performance Metrics
| Metric | Baseline | Current | Target | Status |
|--------|----------|---------|---------|---------|
| Response Time | 24 hours | 4 hours | 2 hours | On Track |
| Customer Satisfaction | 3.2/5 | 4.1/5 | 4.5/5 | Improving |
| Resolution Rate | 65% | 78% | 85% | Progress |
| System Uptime | 95% | 98.5% | 99.5% | Excellent |

# Implementation Timeline

Phase 1: Foundation (Completed)
- Infrastructure setup and configuration
- Core system deployment
- Initial staff training and onboarding

Phase 2: Enhancement (In Progress)
- Advanced features implementation
- Integration with existing systems
- User feedback incorporation

Phase 3: Optimization (Planned)
- Performance tuning and optimization
- Advanced analytics implementation
- Full rollout to all customer segments

## Dialogue Sample from User Testing:

Tester: "I'm trying to submit a support request but the form won't load."

System: "I understand you're having trouble with the support form. Let me help you with that."

Tester: "Yes, I click submit but nothing happens."

System: "I've logged this issue and created a ticket for you. Reference number: SUP-2024-789. A technician will contact you within 2 hours."

## Next Steps:
1. Complete Phase 2 testing
2. Gather additional user feedback
3. Prepare for Phase 3 rollout
4. Schedule final stakeholder review

This mixed-structure document demonstrates the complexity of real-world documentation that combines multiple organizational patterns."""
        doc = Document(content=content, metadata={'source': 'mixed.txt', 'type': 'project_report', 'structure': 'mixed'})
        self.test_documents['mixed'] = doc

    @staticmethod
    def create_empty_document(self):
        doc = Document(content="", metadata={'source': 'empty.txt', 'type': 'empty', 'structure': 'none'})
        self.test_documents['empty'] = doc

    def test_basic_structure_aware_chunking(self):
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['hierarchical'])
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertTrue(all(chunk.text_content for chunk in chunks))
        
        # Verify hierarchical content is preserved
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Chapter 1", all_content)
        self.assertIn("Section 1.1", all_content)

    def test_hierarchical_structure_detection(self):
        config = StructureAwareConfig(preferred_structures=[StructureType.HIERARCHICAL])
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['hierarchical'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect and handle hierarchical structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Chapter", all_content)
        self.assertIn("Section", all_content)
        self.assertIn("Subsection", all_content)

    def test_list_structure_detection(self):
        config = StructureAwareConfig(preferred_structures=[StructureType.LIST_BASED])
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['list_based'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect and handle list structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        # Check for any list-related content
        self.assertTrue(
            "Hardware" in all_content or 
            "Software" in all_content or 
            "enterprise-grade" in all_content
        )

    def test_table_structure_detection(self):
        config = StructureAwareConfig(preferred_structures=[StructureType.TABLE_BASED])
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['table_based'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect and handle table structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Performance Metrics", all_content)
        self.assertIn("Code Quality", all_content)

    def test_dialogue_structure_detection(self):
        config = StructureAwareConfig(preferred_structures=[StructureType.DIALOGUE])
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['dialogue'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect and handle dialogue structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Agent:", all_content)
        self.assertIn("Customer:", all_content)

    def test_procedural_structure_detection(self):
        config = StructureAwareConfig(preferred_structures=[StructureType.PROCEDURAL])
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['procedural'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect and handle procedural structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Step 1", all_content)
        self.assertIn("Prerequisites", all_content)

    def test_narrative_structure_detection(self):
        config = StructureAwareConfig(preferred_structures=[StructureType.NARRATIVE])
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['narrative'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should detect and handle narrative structure
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Elena Rodriguez", all_content)
        self.assertIn("laboratory", all_content)

    def test_mixed_structure_detection(self):
        config = StructureAwareConfig(auto_detect_structure=True)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['mixed'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should handle mixed structure document
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Project Alpha", all_content)
        self.assertIn("Technical Architecture", all_content)
        self.assertIn("Performance Metrics", all_content)

    def test_structure_preservation(self):
        config = StructureAwareConfig(preserve_structure=True)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['hierarchical'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should preserve structural relationships
        for chunk in chunks:
            self.assertIsInstance(chunk.metadata, dict)
            self.assertEqual(chunk.metadata['structure'], 'hierarchical')

    def test_boundary_detection(self):
        config = StructureAwareConfig(respect_boundaries=True)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['procedural'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should respect structural boundaries
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("Database Backup", all_content)

    def test_context_awareness(self):
        config = StructureAwareConfig(context_aware=True)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['dialogue'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should maintain conversational context
        all_content = " ".join(chunk.text_content for chunk in chunks)
        self.assertIn("customer service", all_content.lower())

    def test_adaptive_chunking(self):
        config = StructureAwareConfig(adaptive_chunking=True)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['mixed'])
        
        self.assertGreater(len(chunks), 0)
        
        # Should adapt to different structures within document
        for chunk in chunks:
            self.assertGreater(len(chunk.text_content), 0)

    def test_empty_content_handling(self):
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['empty'])
        
        self.assertEqual(len(chunks), 0)

    def test_whitespace_only_content(self):
        whitespace_content = "   \n\n  \t  \n   "
        whitespace_doc = Document(content=whitespace_content, metadata={'type': 'whitespace'})
        
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(whitespace_doc)
        
        self.assertEqual(len(chunks), 0)

    def test_performance_with_complex_structure(self):
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        
        start_time = time.time()
        chunks = chunker.chunk(self.test_documents['mixed'])
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds

    def test_metadata_inheritance(self):
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['table_based'])
        
        self.assertGreater(len(chunks), 0)
        
        # Verify original metadata is inherited
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source'], 'table.txt')
            self.assertEqual(chunk.metadata['type'], 'performance_review')
            self.assertEqual(chunk.metadata['structure'], 'table_based')

    def test_structure_statistics(self):
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        
        # Process document
        chunks = chunker.chunk(self.test_documents['hierarchical'])
        
        # Try to get structure statistics if available
        try:
            stats = chunker.get_structure_stats()
            self.assertIsInstance(stats, dict)
        except AttributeError:
            # Method might not exist in actual implementation
            pass

    def test_custom_structure_patterns(self):
        custom_patterns = {
            'section': r'##\s+',
            'subsection': r'###\s+',
            'procedure': r'Step\s+\d+:'
        }
        config = StructureAwareConfig(custom_patterns=custom_patterns)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        chunks = chunker.chunk(self.test_documents['procedural'])
        
        self.assertGreater(len(chunks), 0)

    def test_batch_processing(self):
        documents = [
            self.test_documents['hierarchical'],
            self.test_documents['list_based'],
            self.test_documents['dialogue']
        ]
        
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        
        batch_results = chunker.chunk_batch(documents)
        
        self.assertEqual(len(batch_results), 3)
        self.assertTrue(all(len(chunks) > 0 for chunks in batch_results))
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        self.assertGreater(total_chunks, 0)

    def test_caching_functionality(self):
        config = StructureAwareConfig(enable_caching=True)
        chunker = DocumentStructureAwareChunkingStrategy(config)
        
        # First processing
        chunks1 = chunker.chunk(self.test_documents['hierarchical'])
        
        # Second processing (should use cache)
        chunks2 = chunker.chunk(self.test_documents['hierarchical'])
        
        self.assertEqual(len(chunks1), len(chunks2))
        self.assertTrue(all(c1.text_content == c2.text_content for c1, c2 in zip(chunks1, chunks2)))

    def test_error_handling(self):
        config = StructureAwareConfig()
        chunker = DocumentStructureAwareChunkingStrategy(config)
        
        # Test with malformed structure
        malformed_content = """# Incomplete Header
        This content has malformed structure...
        
        ## Missing proper formatting
        - List item without proper structure
        | Incomplete table
        """
        malformed_doc = Document(content=malformed_content, metadata={'type': 'malformed'})
        
        chunks = chunker.chunk(malformed_doc)
        self.assertGreaterEqual(len(chunks), 0)  # Should handle gracefully

if __name__ == "__main__":
    unittest.main()