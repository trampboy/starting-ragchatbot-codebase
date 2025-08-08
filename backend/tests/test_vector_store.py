"""Tests for VectorStore search functionality and data retrieval."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from vector_store import VectorStore, SearchResults
from models import Course, CourseChunk, Lesson
from .test_utils import MockChromaClient, MockChromaCollection, create_temp_chroma_path, cleanup_temp_path


class TestVectorStore:
    """Test cases for VectorStore functionality"""
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client"""
        return MockChromaClient()
    
    @pytest.fixture
    def vector_store_with_mock(self, mock_chroma_client):
        """Create VectorStore with mocked ChromaDB client"""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client_class.return_value = mock_chroma_client
            
            # Mock the embedding function
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                store = VectorStore(
                    chroma_path="./test_chroma", 
                    embedding_model="test-model",
                    max_results=5
                )
                store.client = mock_chroma_client  # Ensure we use our mock
                return store, mock_chroma_client
    
    def setup_sample_data(self, mock_client):
        """Set up sample data in the mock client"""
        # Add sample course catalog data
        catalog = mock_client.collections.get('course_catalog', MockChromaCollection())
        catalog.documents = ["Introduction to Machine Learning", "Advanced Deep Learning"]
        catalog.metadata = [
            {
                'title': 'Introduction to Machine Learning',
                'instructor': 'Dr. Smith',
                'course_link': 'https://example.com/ml',
                'lessons_json': '[{"lesson_number": 0, "lesson_title": "Getting Started", "lesson_link": "https://example.com/ml/0"}]',
                'lesson_count': 1
            },
            {
                'title': 'Advanced Deep Learning',
                'instructor': 'Prof. Johnson', 
                'course_link': 'https://example.com/dl',
                'lessons_json': '[{"lesson_number": 0, "lesson_title": "Neural Basics", "lesson_link": "https://example.com/dl/0"}]',
                'lesson_count': 1
            }
        ]
        catalog.ids = ['Introduction to Machine Learning', 'Advanced Deep Learning']
        
        # Add sample course content data
        content = mock_client.collections.get('course_content', MockChromaCollection())
        content.documents = [
            "Course Introduction to Machine Learning Lesson 0 content: This is an introduction to ML concepts and supervised learning.",
            "Advanced topics in deep learning including neural networks and backpropagation algorithms."
        ]
        content.metadata = [
            {'course_title': 'Introduction to Machine Learning', 'lesson_number': 0, 'chunk_index': 0},
            {'course_title': 'Advanced Deep Learning', 'lesson_number': 0, 'chunk_index': 1}
        ]
        content.ids = ['Introduction_to_Machine_Learning_0', 'Advanced_Deep_Learning_1']
        
        mock_client.collections['course_catalog'] = catalog
        mock_client.collections['course_content'] = content

    @pytest.mark.unit
    def test_search_with_valid_query(self, vector_store_with_mock):
        """Test search with a valid query"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        results = store.search("machine learning concepts")
        
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert results.error is None
    
    @pytest.mark.unit
    def test_search_with_course_name_filter(self, vector_store_with_mock):
        """Test search with course name filter"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        results = store.search("concepts", course_name="Introduction to Machine Learning")
        
        assert not results.is_empty()
        assert results.error is None
        # The search should have triggered course name resolution
    
    @pytest.mark.unit
    def test_search_with_lesson_number_filter(self, vector_store_with_mock):
        """Test search with lesson number filter"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        results = store.search("neural networks", lesson_number=0)
        
        assert not results.is_empty()
        assert results.error is None
    
    @pytest.mark.unit
    def test_search_with_both_filters(self, vector_store_with_mock):
        """Test search with both course name and lesson number filters"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        results = store.search("learning", course_name="Introduction to Machine Learning", lesson_number=0)
        
        assert not results.is_empty()
        assert results.error is None
    
    @pytest.mark.unit
    def test_search_with_invalid_course_name(self, vector_store_with_mock):
        """Test search with non-existent course name"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        # Mock _resolve_course_name to return None for unknown course
        with patch.object(store, '_resolve_course_name', return_value=None):
            results = store.search("concepts", course_name="Non-existent Course")
        
        assert results.error is not None
        assert "No course found matching" in results.error
    
    @pytest.mark.unit
    def test_search_with_database_error(self, vector_store_with_mock):
        """Test search when database throws an error"""
        store, mock_client = vector_store_with_mock
        
        # Make the content collection raise an error
        mock_client.collections['course_content'] = MockChromaCollection()
        mock_client.collections['course_content'].should_raise_error = True
        mock_client.collections['course_content'].error_message = "Database connection failed"
        
        results = store.search("test query")
        
        assert results.error is not None
        assert "Search error" in results.error
    
    @pytest.mark.unit
    def test_resolve_course_name_success(self, vector_store_with_mock):
        """Test successful course name resolution"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        resolved_name = store._resolve_course_name("Machine Learning")
        
        assert resolved_name == "Introduction to Machine Learning"
    
    @pytest.mark.unit
    def test_resolve_course_name_no_match(self, vector_store_with_mock):
        """Test course name resolution with no matches"""
        store, mock_client = vector_store_with_mock
        # Don't set up data - empty catalog
        
        resolved_name = store._resolve_course_name("Non-existent Course")
        
        assert resolved_name is None
    
    @pytest.mark.unit
    def test_build_filter_no_params(self, vector_store_with_mock):
        """Test filter building with no parameters"""
        store, _ = vector_store_with_mock
        
        filter_dict = store._build_filter(None, None)
        
        assert filter_dict is None
    
    @pytest.mark.unit
    def test_build_filter_course_only(self, vector_store_with_mock):
        """Test filter building with course title only"""
        store, _ = vector_store_with_mock
        
        filter_dict = store._build_filter("Test Course", None)
        
        assert filter_dict == {"course_title": "Test Course"}
    
    @pytest.mark.unit
    def test_build_filter_lesson_only(self, vector_store_with_mock):
        """Test filter building with lesson number only"""
        store, _ = vector_store_with_mock
        
        filter_dict = store._build_filter(None, 1)
        
        assert filter_dict == {"lesson_number": 1}
    
    @pytest.mark.unit
    def test_build_filter_both_params(self, vector_store_with_mock):
        """Test filter building with both parameters"""
        store, _ = vector_store_with_mock
        
        filter_dict = store._build_filter("Test Course", 1)
        
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 1}
        ]}
        assert filter_dict == expected
    
    @pytest.mark.unit
    def test_add_course_metadata(self, vector_store_with_mock):
        """Test adding course metadata to catalog"""
        store, mock_client = vector_store_with_mock
        
        course = Course(
            title="Test Course",
            course_link="https://test.com",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Intro", lesson_link="https://test.com/0"),
                Lesson(lesson_number=1, title="Advanced", lesson_link="https://test.com/1")
            ]
        )
        
        store.add_course_metadata(course)
        
        catalog = mock_client.collections['course_catalog']
        assert "Test Course" in catalog.documents
        assert len(catalog.metadata) == 1
        assert catalog.metadata[0]['title'] == 'Test Course'
        assert catalog.metadata[0]['instructor'] == 'Test Instructor'
        assert 'lessons_json' in catalog.metadata[0]
    
    @pytest.mark.unit
    def test_add_course_content(self, vector_store_with_mock):
        """Test adding course content chunks"""
        store, mock_client = vector_store_with_mock
        
        chunks = [
            CourseChunk(
                content="Test content 1",
                course_title="Test Course",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Test content 2", 
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        store.add_course_content(chunks)
        
        content = mock_client.collections['course_content']
        assert len(content.documents) == 2
        assert "Test content 1" in content.documents
        assert "Test content 2" in content.documents
    
    @pytest.mark.unit
    def test_get_existing_course_titles(self, vector_store_with_mock):
        """Test retrieving existing course titles"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        titles = store.get_existing_course_titles()
        
        assert "Introduction to Machine Learning" in titles
        assert "Advanced Deep Learning" in titles
        assert len(titles) == 2
    
    @pytest.mark.unit
    def test_get_course_count(self, vector_store_with_mock):
        """Test getting course count"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        count = store.get_course_count()
        
        assert count == 2
    
    @pytest.mark.unit
    def test_get_lesson_link(self, vector_store_with_mock):
        """Test getting lesson link for course and lesson"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        link = store.get_lesson_link("Introduction to Machine Learning", 0)
        
        assert link == "https://example.com/ml/0"
    
    @pytest.mark.unit
    def test_get_lesson_link_not_found(self, vector_store_with_mock):
        """Test getting lesson link when not found"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        link = store.get_lesson_link("Non-existent Course", 0)
        
        assert link is None
    
    @pytest.mark.unit
    def test_clear_all_data(self, vector_store_with_mock):
        """Test clearing all data"""
        store, mock_client = vector_store_with_mock
        self.setup_sample_data(mock_client)
        
        # Verify data exists
        assert len(mock_client.collections) > 0
        
        store.clear_all_data()
        
        # Collections should be recreated (empty)
        assert 'course_catalog' in mock_client.collections
        assert 'course_content' in mock_client.collections


class TestSearchResults:
    """Test cases for SearchResults class"""
    
    @pytest.mark.unit
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    @pytest.mark.unit
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    @pytest.mark.unit
    def test_empty_results_with_error(self):
        """Test creating empty results with error message"""
        error_msg = "Database connection failed"
        
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
    
    @pytest.mark.unit
    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        
        assert results.is_empty() is True
    
    @pytest.mark.unit
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=['doc1'], 
            metadata=[{'key': 'value'}], 
            distances=[0.1]
        )
        
        assert results.is_empty() is False