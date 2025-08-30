from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import re
import time
from enum import Enum
from pydantic import Field

from .base import ChunkingStrategy, ChunkingConfig
from ..schemas.data_models import Document, Chunk


class StructureType(Enum):
    """Types of document structures that can be detected."""
    HIERARCHICAL = "hierarchical"
    LIST_BASED = "list_based"
    TABLE_BASED = "table_based"
    DIALOGUE = "dialogue"
    PROCEDURAL = "procedural"
    NARRATIVE = "narrative"
    MIXED = "mixed"


class StructuralElement(NamedTuple):
    """Represents a structural element in the document."""
    type: str
    level: int
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]


class StructureAwareConfig(ChunkingConfig):
    """Configuration for structure-aware chunking."""
    
    detect_headers: bool = Field(True, description="Detect and preserve header hierarchy")
    detect_lists: bool = Field(True, description="Detect and preserve list structures")
    detect_tables: bool = Field(True, description="Detect and preserve table structures")
    detect_code_blocks: bool = Field(True, description="Detect and preserve code blocks")
    detect_quotes: bool = Field(True, description="Detect and preserve quotes/citations")
    
    preserve_hierarchy: bool = Field(True, description="Maintain hierarchical relationships")
    max_hierarchy_depth: int = Field(6, description="Maximum hierarchy depth to track")
    include_parent_context: bool = Field(True, description="Include parent section context")
    
    keep_lists_together: bool = Field(True, description="Try to keep list items together")
    keep_tables_together: bool = Field(True, description="Try to keep tables together")
    keep_code_together: bool = Field(True, description="Try to keep code blocks together")
    
    detect_references: bool = Field(True, description="Detect cross-references and maintain context")
    link_related_sections: bool = Field(True, description="Link related sections")
    
    min_section_size: int = Field(100, description="Minimum size for a section to be chunked separately")
    max_section_size: int = Field(5000, description="Maximum size before splitting a section")
    balance_chunk_sizes: bool = Field(True, description="Try to balance chunk sizes")


class DocumentStructureAnalyzer:
    """Analyzes document structure for intelligent chunking."""
    
    def __init__(self, config: StructureAwareConfig):
        self.config = config
        
        self.header_patterns = [
            r'^(#{1,6})\s+(.+)$',
            r'^(\d+\.)+\s+(.+)$',
            r'^([IVXLCDM]+\.)\s+(.+)$',
            r'^([A-Z]\.)\s+(.+)$',
        ]
        
        self.list_patterns = [
            r'^\s*[-*+]\s+(.+)$',
            r'^\s*\d+\.\s+(.+)$',
            r'^\s*[a-zA-Z]\.\s+(.+)$',
            r'^\s*[ivx]+\.\s+(.+)$',
        ]
        
        self.table_patterns = [
            r'^\s*\|.*\|$',
            r'^\s*[-+]+\s*$',
        ]
        
        self.code_patterns = [
            r'```[\s\S]*?```',
            r'`[^`]+`',
            r'^\s{4,}.+$',
        ]
        
        self.quote_patterns = [
            r'^>\s+(.+)$',
            r'^["""].+["""]$',
        ]
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze document structure comprehensively."""
        analysis = {
            'structure_type': self._detect_overall_structure(content),
            'elements': self._extract_structural_elements(content),
            'hierarchy': self._build_hierarchy(content),
            'relationships': self._detect_relationships(content),
            'statistics': self._calculate_statistics(content)
        }
        
        return analysis
    
    def _detect_overall_structure(self, content: str) -> StructureType:
        """Detect the overall structure type of the document."""
        structure_scores = {
            StructureType.HIERARCHICAL: 0,
            StructureType.LIST_BASED: 0,
            StructureType.TABLE_BASED: 0,
            StructureType.DIALOGUE: 0,
            StructureType.PROCEDURAL: 0,
            StructureType.NARRATIVE: 0
        }
        
        lines = content.split('\n')
        
        header_count = sum(1 for line in lines if any(re.match(pattern, line) for pattern in self.header_patterns))
        structure_scores[StructureType.HIERARCHICAL] = header_count / max(len(lines), 1)
        
        step_indicators = ['step', 'first', 'then', 'next', 'finally']
        step_count = sum(content.lower().count(indicator) for indicator in step_indicators)
        
        procedure_pattern = r'\b(?:step|first|then|next|finally)\s+\d+.*procedure\b'
        procedure_matches = len(re.findall(procedure_pattern, content.lower()))
        step_count += procedure_matches
        
        if step_count > 0:
            structure_scores[StructureType.PROCEDURAL] = max(0.8, step_count / max(len(content), 1))
        else:
            structure_scores[StructureType.PROCEDURAL] = step_count / max(len(content), 1)
        
        list_count = sum(1 for line in lines if any(re.match(pattern, line) for pattern in self.list_patterns))
        structure_scores[StructureType.LIST_BASED] = list_count / max(len(lines), 1)
        
        table_count = sum(1 for line in lines if any(re.match(pattern, line) for pattern in self.table_patterns))
        structure_scores[StructureType.TABLE_BASED] = table_count / max(len(lines), 1)
        
        qa_indicators = ['?', 'Q:', 'A:', 'Question:', 'Answer:']
        qa_count = sum(content.lower().count(indicator.lower()) for indicator in qa_indicators)
        structure_scores[StructureType.DIALOGUE] = qa_count / max(len(content), 1)
        
        narrative_indicators = ['story', 'once', 'chapter', 'paragraph', 'narrative']
        narrative_count = sum(content.lower().count(indicator) for indicator in narrative_indicators)
        structure_scores[StructureType.NARRATIVE] = narrative_count / max(len(content), 1)
        
        max_score = max(structure_scores.values())
        if max_score < 0.1:
            return StructureType.MIXED
        
        if structure_scores[StructureType.PROCEDURAL] > 0.5:
            return StructureType.PROCEDURAL
        
        significant_structures = sum(1 for score in structure_scores.values() if score > 0.05)
        if significant_structures > 2:
            return StructureType.MIXED
        
        return max(structure_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_structural_elements(self, content: str) -> List[StructuralElement]:
        """Extract all structural elements from the document."""
        elements = []
        lines = content.split('\n')
        current_pos = 0
        
        for line_num, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)
            line_stripped = line.strip()
            
            if not line_stripped:
                current_pos = line_end + 1
                continue
            
            element_added = False
            
            if self.config.detect_headers and not element_added:
                for pattern in self.header_patterns:
                    match = re.match(pattern, line_stripped)
                    if match:
                        level = len(match.group(1)) if match.group(1).startswith('#') else 1
                        elements.append(StructuralElement(
                            type='header',
                            level=level,
                            text=line_stripped,
                            start_pos=line_start,
                            end_pos=line_end,
                            metadata={'pattern': pattern, 'line_num': line_num}
                        ))
                        element_added = True
                        break
            
            if self.config.detect_lists and not element_added:
                for pattern in self.list_patterns:
                    match = re.match(pattern, line)
                    if match:
                        elements.append(StructuralElement(
                            type='list_item',
                            level=len(line) - len(line.lstrip()),
                            text=line_stripped,
                            start_pos=line_start,
                            end_pos=line_end,
                            metadata={'pattern': pattern, 'line_num': line_num}
                        ))
                        element_added = True
                        break
            
            if self.config.detect_tables and not element_added:
                for pattern in self.table_patterns:
                    if re.match(pattern, line):
                        elements.append(StructuralElement(
                            type='table_row',
                            level=0,
                            text=line_stripped,
                            start_pos=line_start,
                            end_pos=line_end,
                            metadata={'pattern': pattern, 'line_num': line_num}
                        ))
                        element_added = True
                        break
            
            current_pos = line_end + 1
        
        if self.config.detect_code_blocks:
            for pattern in self.code_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    elements.append(StructuralElement(
                        type='code_block',
                        level=0,
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        metadata={'pattern': pattern}
                    ))
        
        elements.sort(key=lambda x: x.start_pos)
        return elements
    
    def _build_hierarchy(self, content: str) -> Dict[str, Any]:
        """Build hierarchical structure from detected elements."""
        elements = self._extract_structural_elements(content)
        hierarchy = {'root': {'children': [], 'level': 0, 'elements': []}}
        
        current_stack = [hierarchy['root']]
        
        for element in elements:
            if element.type == 'header':
                while len(current_stack) > 1 and current_stack[-1]['level'] >= element.level:
                    current_stack.pop()
                
                node = {
                    'element': element,
                    'children': [],
                    'level': element.level,
                    'elements': []
                }
                
                current_stack[-1]['children'].append(node)
                current_stack.append(node)
            else:
                if current_stack:
                    current_stack[-1]['elements'].append(element)
        
        return hierarchy
    
    def _detect_relationships(self, content: str) -> Dict[str, List[Tuple[int, int]]]:
        """Detect relationships between different parts of the document."""
        relationships = {
            'references': [],
            'continuations': [],
            'examples': []
        }
        
        reference_patterns = [
            r'see\s+(?:section|chapter|part)\s+(\d+)',
            r'as\s+(?:mentioned|discussed|shown)\s+(?:in|above|below)',
            r'refer\s+to\s+(?:section|chapter|appendix)\s+(\w+)'
        ]
        
        for pattern in reference_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                relationships['references'].append((match.start(), match.end()))
        
        return relationships
    
    def _calculate_statistics(self, content: str) -> Dict[str, Any]:
        """Calculate document structure statistics."""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1),
            'header_count': sum(1 for line in lines if any(re.match(pattern, line.strip()) for pattern in self.header_patterns)),
            'list_count': sum(1 for line in lines if any(re.match(pattern, line) for pattern in self.list_patterns)),
            'code_block_count': len(re.findall(r'```[\s\S]*?```', content)),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        }


class DocumentStructureAwareChunkingStrategy(ChunkingStrategy):
    """
    Advanced document structure-aware chunking strategy.
    
    This strategy analyzes document structure and creates chunks that respect
    the logical organization of content. It preserves hierarchies, maintains
    relationships between sections, and ensures that each chunk represents
    a coherent logical unit.
    
    Features:
    - Automatic structure detection and analysis
    - Hierarchy-preserving chunking
    - Context preservation across related sections
    - Adaptive chunk sizing based on content structure
    - Support for multiple document types and structures
    - Relationship detection and maintenance
    
    Examples:
        # Basic usage
        strategy = DocumentStructureAwareChunkingStrategy()
        
        # Advanced configuration
        config = StructureAwareConfig(
            chunk_size=1200,
            preserve_hierarchy=True,
            include_parent_context=True,
            keep_lists_together=True,
            detect_references=True
        )
        strategy = DocumentStructureAwareChunkingStrategy(config=config)
    """
    
    def __init__(self, config: Optional[StructureAwareConfig] = None):
        """Initialize structure-aware chunking strategy."""
        if config is None:
            config = StructureAwareConfig()
        
        super().__init__(config)
        
        self.analyzer = DocumentStructureAnalyzer(self.config)
        
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Create structure-aware chunks that respect document organization.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of structure-aware chunks
        """
        start_time = time.time()
        
        if not document.content.strip():
            return []
        
        structure_analysis = self.analyzer.analyze(document.content)
        
        structural_chunks = self._create_structural_chunks(
            document, structure_analysis
        )
        
        final_chunks = self._optimize_structural_chunks(
            structural_chunks, document, structure_analysis
        )
        
        processing_time = (time.time() - start_time) * 1000
        self._update_metrics(final_chunks, processing_time, document)
        
        return final_chunks
    
    def _create_structural_chunks(
        self, 
        document: Document, 
        analysis: Dict[str, Any]
    ) -> List[Chunk]:
        """Create chunks based on document structure analysis."""
        content = document.content
        structure_type = analysis['structure_type']
        elements = analysis['elements']
        hierarchy = analysis['hierarchy']
        
        if structure_type == StructureType.HIERARCHICAL:
            chunks = self._create_hierarchical_chunks(document, hierarchy, elements)
        elif structure_type == StructureType.LIST_BASED:
            chunks = self._create_list_based_chunks(document, elements)
        elif structure_type == StructureType.TABLE_BASED:
            chunks = self._create_table_based_chunks(document, elements)
        elif structure_type == StructureType.PROCEDURAL:
            chunks = self._create_procedural_chunks(document, elements)
        else:
            chunks = self._create_mixed_chunks(document, analysis)
        
        if not chunks:
            chunks = self._create_simple_chunks(document)
        
        return chunks
    
    def _create_hierarchical_chunks(
        self, 
        document: Document, 
        hierarchy: Dict[str, Any], 
        elements: List[StructuralElement]
    ) -> List[Chunk]:
        """Create chunks respecting hierarchical structure."""
        chunks = []
        
        current_section = []
        current_header = None
        
        for element in elements:
            if element.type == 'header':
                if current_section and current_header:
                    section_content = current_header.text + "\n" + "\n".join([e.text for e in current_section])
                    chunk = self._create_chunk(
                        text_content=section_content,
                        document=document,
                        chunk_index=len(chunks),
                        total_chunks=1
                    )
                    
                    chunk.metadata.update({
                        'section_type': 'hierarchical',
                        'header_level': current_header.level,
                        'section_elements': len(current_section)
                    })
                    
                    chunks.append(chunk)
                
                current_header = element
                current_section = []
            else:
                current_section.append(element)
        
        if current_section and current_header:
            section_content = current_header.text + "\n" + "\n".join([e.text for e in current_section])
            chunk = self._create_chunk(
                text_content=section_content,
                document=document,
                chunk_index=len(chunks),
                total_chunks=1
            )
            
            chunk.metadata.update({
                'section_type': 'hierarchical',
                'header_level': current_header.level,
                'section_elements': len(current_section)
            })
            
            chunks.append(chunk)
        
        if not chunks and elements:
            for element in elements:
                if element.type == 'header':
                    chunk = self._create_chunk(
                        text_content=element.text,
                        document=document,
                        chunk_index=len(chunks),
                        total_chunks=1
                    )
                    
                    chunk.metadata.update({
                        'section_type': 'hierarchical',
                        'header_level': element.level,
                        'section_elements': 0
                    })
                    
                    chunks.append(chunk)
        
        return chunks
    
    def _create_list_based_chunks(self, document: Document, elements: List[StructuralElement]) -> List[Chunk]:
        """Create chunks optimized for list-based content."""
        chunks = []
        current_list = []
        current_size = 0
        
        content_pos = 0
        for element in elements:
            if element.type == 'list_item':
                item_size = len(element.text)
                
                if (current_size + item_size > self.config.chunk_size and 
                    current_list and 
                    not self.config.keep_lists_together):
                    
                    list_content = '\n'.join(current_list)
                    chunk = self._create_chunk(
                        text_content=list_content,
                        document=document,
                        chunk_index=len(chunks),
                        total_chunks=1
                    )
                    
                    chunk.metadata.update({
                        'content_type': 'list',
                        'list_item_count': len(current_list)
                    })
                    
                    chunks.append(chunk)
                    
                    current_list = []
                    current_size = 0
                
                current_list.append(element.text)
                current_size += item_size
        
        # Create final chunk
        if current_list:
            list_content = '\n'.join(current_list)
            chunk = self._create_chunk(
                text_content=list_content,
                document=document,
                chunk_index=len(chunks),
                total_chunks=1
            )
            
            chunk.metadata.update({
                'content_type': 'list',
                'list_item_count': len(current_list)
            })
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_table_based_chunks(self, document: Document, elements: List[StructuralElement]) -> List[Chunk]:
        """Create chunks optimized for tabular content."""
        chunks = []
        current_table = []
        
        for element in elements:
            if element.type == 'table_row':
                current_table.append(element.text)
                
                table_size = sum(len(row) for row in current_table)
                if (table_size > self.config.chunk_size and 
                    len(current_table) > 1 and 
                    not self.config.keep_tables_together):
                    
                    table_content = '\n'.join(current_table[:-1])
                    chunk = self._create_chunk(
                        text_content=table_content,
                        document=document,
                        chunk_index=len(chunks),
                        total_chunks=1
                    )
                    
                    chunk.metadata.update({
                        'content_type': 'table',
                        'row_count': len(current_table) - 1
                    })
                    
                    chunks.append(chunk)
                    
                    current_table = [current_table[-1]]
        
        # Create final table chunk
        if current_table:
            table_content = '\n'.join(current_table)
            chunk = self._create_chunk(
                text_content=table_content,
                document=document,
                chunk_index=len(chunks),
                total_chunks=1
            )
            
            chunk.metadata.update({
                'content_type': 'table',
                'row_count': len(current_table)
            })
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_procedural_chunks(self, document: Document, elements: List[StructuralElement]) -> List[Chunk]:
        """Create chunks optimized for procedural content."""
        content = document.content
        
        step_pattern = r'\n(?=(?:step\s+\d+|^\d+\.|first|then|next|finally))'
        steps = re.split(step_pattern, content, flags=re.IGNORECASE | re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        
        for step in steps:
            step = step.strip()
            if not step:
                continue
            
            if len(current_chunk) + len(step) > self.config.chunk_size and current_chunk:
                chunk = self._create_chunk(
                    text_content=current_chunk.strip(),
                    document=document,
                    chunk_index=len(chunks),
                    total_chunks=1
                )
                
                chunk.metadata.update({
                    'content_type': 'procedural',
                    'step_based': True
                })
                
                chunks.append(chunk)
                current_chunk = ""
            
            current_chunk += step + "\n\n"
        
        # Create final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                text_content=current_chunk.strip(),
                document=document,
                chunk_index=len(chunks),
                total_chunks=1
            )
            
            chunk.metadata.update({
                'content_type': 'procedural',
                'step_based': True
            })
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_mixed_chunks(self, document: Document, analysis: Dict[str, Any]) -> List[Chunk]:
        """Create chunks for mixed structure documents."""
        elements = analysis['elements']
        content = document.content
        
        structure_regions = self._identify_structure_regions(elements, content)
        
        chunks = []
        for region in structure_regions:
            region_chunks = self._chunk_region(region, document)
            chunks.extend(region_chunks)
        
        return chunks
    
    def _identify_structure_regions(self, elements: List[StructuralElement], content: str) -> List[Dict[str, Any]]:
        """Identify regions with consistent structure types."""
        regions = []
        current_region = None
        
        for element in elements:
            if current_region is None or current_region['type'] != element.type:
                if current_region:
                    regions.append(current_region)
                
                current_region = {
                    'type': element.type,
                    'start_pos': element.start_pos,
                    'end_pos': element.end_pos,
                    'elements': [element]
                }
            else:
                current_region['end_pos'] = element.end_pos
                current_region['elements'].append(element)
        
        if current_region:
            regions.append(current_region)
        
        return regions
    
    def _chunk_region(self, region: Dict[str, Any], document: Document) -> List[Chunk]:
        """Chunk a specific structural region."""
        region_type = region['type']
        elements = region['elements']
        
        start_pos = region['start_pos']
        end_pos = region['end_pos']
        region_content = document.content[start_pos:end_pos]
        
        chunk = self._create_chunk(
            text_content=region_content.strip(),
            document=document,
            chunk_index=0,
            total_chunks=1,
            start_pos=start_pos,
            end_pos=end_pos
        )
        
        chunk.metadata.update({
            'region_type': region_type,
            'element_count': len(elements),
            'structure_preserved': True
        })
        
        if region_type == 'header':
            chunk.metadata['section_type'] = 'hierarchical'
        elif region_type == 'list_item':
            chunk.metadata['content_type'] = 'list'
        elif region_type == 'table_row':
            chunk.metadata['content_type'] = 'table'
        elif region_type == 'code_block':
            chunk.metadata['content_type'] = 'code'
        
        return [chunk]
    
    def _optimize_structural_chunks(
        self, 
        chunks: List[Chunk], 
        document: Document, 
        analysis: Dict[str, Any]
    ) -> List[Chunk]:
        """Optimize structural chunks for better quality and balance."""
        if not chunks:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            if len(chunk.text_content) < self.config.min_section_size:
                if (optimized_chunks and 
                    len(optimized_chunks[-1].text_content) + len(chunk.text_content) <= self.config.max_section_size):
                    
                    prev_chunk = optimized_chunks[-1]
                    merged_content = prev_chunk.text_content + "\n\n" + chunk.text_content
                    prev_chunk.text_content = merged_content
                    prev_chunk.metadata['merged_small_chunk'] = True
                    continue
            
            if len(chunk.text_content) > self.config.max_section_size:
                split_chunks = self._split_large_structural_chunk(chunk, document)
                optimized_chunks.extend(split_chunks)
            else:
                optimized_chunks.append(chunk)
        
        for i, chunk in enumerate(optimized_chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(optimized_chunks)
        
        return optimized_chunks
    
    def _split_large_structural_chunk(self, chunk: Chunk, document: Document) -> List[Chunk]:
        """Split a large chunk while preserving structural boundaries."""
        content = chunk.text_content
        target_size = self.config.chunk_size
        
        split_points = []
        
        for match in re.finditer(r'\n\s*\n', content):
            split_points.append(match.end())
        
        if not split_points:
            for match in re.finditer(r'[.!?]+\s+', content):
                split_points.append(match.end())
        
        if not split_points:
            for i in range(target_size, len(content), target_size):
                split_points.append(i)
        
        sub_chunks = []
        start = 0
        
        for split_point in split_points:
            if split_point - start >= self.config.min_section_size:
                sub_content = content[start:split_point].strip()
                if sub_content:
                    sub_chunk = self._create_chunk(
                        text_content=sub_content,
                        document=document,
                        chunk_index=len(sub_chunks),
                        total_chunks=1
                    )
                    
                    sub_chunk.metadata.update(chunk.metadata)
                    sub_chunk.metadata['split_from_large'] = True
                    sub_chunk.metadata['original_chunk_size'] = len(content)
                    
                    sub_chunks.append(sub_chunk)
                
                start = split_point
        
        if start < len(content):
            remaining_content = content[start:].strip()
            if remaining_content:
                sub_chunk = self._create_chunk(
                    text_content=remaining_content,
                    document=document,
                    chunk_index=len(sub_chunks),
                    total_chunks=1
                )
                
                sub_chunk.metadata.update(chunk.metadata)
                sub_chunk.metadata['split_from_large'] = True
                sub_chunk.metadata['original_chunk_size'] = len(content)
                
                sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _create_simple_chunks(self, document: Document) -> List[Chunk]:
        """Create simple chunks when structure-based chunking fails."""
        content = document.content
        chunks = []
        
        for i in range(0, len(content), self.config.chunk_size):
            chunk_text = content[i:i + self.config.chunk_size]
            if chunk_text.strip():
                chunk = self._create_chunk(
                    text_content=chunk_text.strip(),
                    document=document,
                    chunk_index=len(chunks),
                    total_chunks=len(content) // self.config.chunk_size + 1,
                    start_pos=i,
                    end_pos=i + len(chunk_text)
                )
                
                chunk.metadata.update({
                    'chunking_method': 'simple_fallback',
                    'structure_type': 'unknown'
                })
                
                chunks.append(chunk)
        
        return chunks
    
    def get_structure_analysis(self, document: Document) -> Dict[str, Any]:
        """Get detailed structure analysis for a document."""
        return self.analyzer.analyze(document.content)
    
    def get_structure_stats(self) -> Dict[str, Any]:
        """Get statistics about structure detection and processing."""
        return {
            "config": {
                "detect_headers": self.config.detect_headers,
                "detect_lists": self.config.detect_lists,
                "detect_tables": self.config.detect_tables,
                "preserve_hierarchy": self.config.preserve_hierarchy,
                "include_parent_context": self.config.include_parent_context
            },
            "cache_size": len(self._analysis_cache),
            "analyzer_patterns": {
                "header_patterns": len(self.analyzer.header_patterns),
                "list_patterns": len(self.analyzer.list_patterns),
                "table_patterns": len(self.analyzer.table_patterns)
            }
        }
