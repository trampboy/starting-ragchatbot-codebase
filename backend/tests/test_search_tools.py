"""Tests for CourseSearchTool.execute() method and search functionality."""

import pytest
from unittest.mock import Mock, patch
import json

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults
from models import Course, Lesson


class TestCourseSearchTool:
    """Test cases for CourseSearchTool functionality"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        mock_store = Mock()
        return mock_store
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool with mock vector store"""
        return CourseSearchTool(mock_vector_store)
    
    @pytest.mark.unit
    def test_get_tool_definition(self, search_tool):
        """Test tool definition is correctly formatted"""
        definition = search_tool.get_tool_definition()
        
        assert definition["type"] == "function"
        assert "function" in definition
        assert definition["function"]["name"] == "search_course_content"
        assert "parameters" in definition["function"]
        assert "query" in definition["function"]["parameters"]["properties"]
        assert "query" in definition["function"]["parameters"]["required"]
    
    @pytest.mark.unit
    def test_execute_successful_search(self, search_tool, mock_vector_store):
        """Test successful search execution"""
        # Setup mock search results
        mock_results = SearchResults(
            documents=[
                "Course Machine Learning Lesson 1 content: Supervised learning uses labeled data.",
                "Advanced concepts in neural networks and deep learning."
            ],
            metadata=[
                {'course_title': 'Machine Learning', 'lesson_number': 1, 'chunk_index': 0},
                {'course_title': 'Deep Learning', 'lesson_number': 2, 'chunk_index': 1}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = search_tool.execute("supervised learning")
        
        assert isinstance(result, str)
        assert "Machine Learning" in result
        assert "Supervised learning uses labeled data" in result
        assert "[Machine Learning - Lesson 1]" in result
        mock_vector_store.search.assert_called_once_with(
            query="supervised learning",
            course_name=None,
            lesson_number=None
        )
    
    @pytest.mark.unit
    def test_execute_with_course_name_filter(self, search_tool, mock_vector_store):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Course content about machine learning"],
            metadata=[{'course_title': 'Machine Learning 101', 'lesson_number': 0}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        result = search_tool.execute("algorithms", course_name="Machine Learning 101")
        
        mock_vector_store.search.assert_called_once_with(
            query="algorithms",
            course_name="Machine Learning 101",
            lesson_number=None
        )
        assert "Machine Learning 101" in result
    
    @pytest.mark.unit
    def test_execute_with_lesson_number_filter(self, search_tool, mock_vector_store):
        """Test search execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 2 covers advanced topics"],
            metadata=[{'course_title': 'Advanced Course', 'lesson_number': 2}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"
        
        result = search_tool.execute("advanced topics", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="advanced topics",
            course_name=None,
            lesson_number=2
        )
        assert "Lesson 2" in result
        assert "advanced topics" in result
    
    @pytest.mark.unit
    def test_execute_with_all_filters(self, search_tool, mock_vector_store):
        """Test search execution with both course name and lesson number"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{'course_title': 'Test Course', 'lesson_number': 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/test/lesson1"
        
        result = search_tool.execute(
            "specific content",
            course_name="Test Course",
            lesson_number=1
        )
        
        mock_vector_store.search.assert_called_once_with(
            query="specific content",
            course_name="Test Course", 
            lesson_number=1
        )
        assert "Test Course - Lesson 1" in result
        assert "specific content" in result
    
    @pytest.mark.unit
    def test_execute_search_error(self, search_tool, mock_vector_store):
        """Test search execution with error from vector store"""
        mock_results = SearchResults.empty("Database connection failed")
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("test query")
        
        assert result == "Database connection failed"
    
    @pytest.mark.unit
    def test_execute_no_results_found(self, search_tool, mock_vector_store):
        """Test search execution with no results"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("nonexistent content")
        
        assert result == "No relevant content found."
    
    @pytest.mark.unit
    def test_execute_no_results_with_filters(self, search_tool, mock_vector_store):
        """Test search execution with no results and filters"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute(
            "nonexistent",
            course_name="Test Course",
            lesson_number=1
        )
        
        expected = "No relevant content found in course 'Test Course' in lesson 1."
        assert result == expected
    
    @pytest.mark.unit
    def test_format_results_with_lesson_links(self, search_tool, mock_vector_store):
        """Test result formatting includes lesson links"""
        mock_results = SearchResults(
            documents=["Content about machine learning"],
            metadata=[{'course_title': 'ML Course', 'lesson_number': 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/ml/lesson1"
        
        formatted = search_tool._format_results(mock_results)
        
        assert "[ML Course - Lesson 1]" in formatted
        assert "Content about machine learning" in formatted
        # Check that sources were tracked with lesson link
        assert len(search_tool.last_sources) == 1
        assert "ML Course - Lesson 1|https://example.com/ml/lesson1" in search_tool.last_sources
    
    @pytest.mark.unit
    def test_format_results_without_lesson_number(self, search_tool, mock_vector_store):
        """Test result formatting when no lesson number is present"""
        mock_results = SearchResults(
            documents=["General course content"],
            metadata=[{'course_title': 'General Course', 'lesson_number': None}],
            distances=[0.1]
        )
        
        formatted = search_tool._format_results(mock_results)
        
        assert "[General Course]" in formatted
        assert "General course content" in formatted
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0] == "General Course"
    
    @pytest.mark.unit
    def test_sources_tracking_reset(self, search_tool, mock_vector_store):
        """Test that sources are properly tracked and can be reset"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{'course_title': 'Test', 'lesson_number': 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://test.com/1"
        
        # Execute search to populate sources
        search_tool.execute("test")
        assert len(search_tool.last_sources) == 1
        
        # Reset sources
        search_tool.last_sources = []
        assert len(search_tool.last_sources) == 0


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool functionality"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        mock_store = Mock()
        return mock_store
    
    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create CourseOutlineTool with mock vector store"""
        return CourseOutlineTool(mock_vector_store)
    
    @pytest.mark.unit
    def test_get_tool_definition(self, outline_tool):
        """Test tool definition is correctly formatted"""
        definition = outline_tool.get_tool_definition()
        
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_course_outline"
        assert "course_name" in definition["function"]["parameters"]["properties"]
        assert "course_name" in definition["function"]["parameters"]["required"]
    
    @pytest.mark.unit
    def test_execute_successful_outline(self, outline_tool, mock_vector_store):
        """Test successful course outline retrieval"""
        # Mock course name resolution
        mock_vector_store._resolve_course_name.return_value = "Machine Learning Course"
        
        # Mock course catalog get response
        lessons_data = [
            {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://example.com/0"},
            {"lesson_number": 1, "lesson_title": "Supervised Learning", "lesson_link": "https://example.com/1"}
        ]
        
        mock_results = {
            'metadatas': [{
                'course_link': 'https://example.com/course',
                'instructor': 'Dr. Smith',
                'lessons_json': json.dumps(lessons_data)
            }]
        }
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        result = outline_tool.execute("Machine Learning")
        
        assert "**Course Title:** Machine Learning Course" in result
        assert "**Course Link:** https://example.com/course" in result
        assert "**Instructor:** Dr. Smith" in result
        assert "**Total Lessons:** 2" in result
        assert "- Lesson 0: Introduction" in result
        assert "- Lesson 1: Supervised Learning" in result
    
    @pytest.mark.unit
    def test_execute_course_not_found(self, outline_tool, mock_vector_store):
        """Test course outline with non-existent course"""
        mock_vector_store._resolve_course_name.return_value = None
        
        result = outline_tool.execute("Non-existent Course")
        
        assert result == "No course found matching 'Non-existent Course'"
    
    @pytest.mark.unit
    def test_execute_no_metadata(self, outline_tool, mock_vector_store):
        """Test course outline when no metadata is found"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.return_value = {'metadatas': []}
        
        result = outline_tool.execute("Test Course")
        
        assert "No metadata found for course 'Test Course'" in result
    
    @pytest.mark.unit
    def test_execute_database_error(self, outline_tool, mock_vector_store):
        """Test course outline with database error"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        result = outline_tool.execute("Test Course")
        
        assert "Error retrieving course outline" in result
        assert "Database error" in result


class TestToolManager:
    """Test cases for ToolManager functionality"""
    
    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance"""
        return ToolManager()
    
    @pytest.fixture
    def mock_search_tool(self, mock_vector_store):
        """Create a mock search tool"""
        return CourseSearchTool(mock_vector_store)
    
    @pytest.fixture
    def mock_outline_tool(self, mock_vector_store):
        """Create a mock outline tool"""  
        return CourseOutlineTool(mock_vector_store)
    
    @pytest.mark.unit
    def test_register_tool(self, tool_manager, mock_search_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_search_tool)
        
        assert "search_course_content" in tool_manager.tools
        assert tool_manager.tools["search_course_content"] == mock_search_tool
    
    @pytest.mark.unit
    def test_register_multiple_tools(self, tool_manager, mock_search_tool, mock_outline_tool):
        """Test registering multiple tools"""
        tool_manager.register_tool(mock_search_tool)
        tool_manager.register_tool(mock_outline_tool)
        
        assert len(tool_manager.tools) == 2
        assert "search_course_content" in tool_manager.tools
        assert "get_course_outline" in tool_manager.tools
    
    @pytest.mark.unit
    def test_get_tool_definitions(self, tool_manager, mock_search_tool, mock_outline_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_search_tool)
        tool_manager.register_tool(mock_outline_tool)
        
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 2
        tool_names = [def_["function"]["name"] for def_ in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    @pytest.mark.unit
    def test_execute_tool_success(self, tool_manager, mock_vector_store):
        """Test successful tool execution"""
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Mock the search to return results
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{'course_title': 'Test', 'lesson_number': 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        result = tool_manager.execute_tool("search_course_content", query="test")
        
        assert isinstance(result, str)
        assert "Test" in result
    
    @pytest.mark.unit
    def test_execute_tool_not_found(self, tool_manager):
        """Test executing non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    @pytest.mark.unit
    def test_get_last_sources(self, tool_manager, mock_vector_store):
        """Test getting sources from last search"""
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = ["Test Course - Lesson 1"]
        tool_manager.register_tool(search_tool)
        
        sources = tool_manager.get_last_sources()
        
        assert sources == ["Test Course - Lesson 1"]
    
    @pytest.mark.unit
    def test_get_last_sources_no_sources(self, tool_manager, mock_vector_store):
        """Test getting sources when no tools have sources"""
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        sources = tool_manager.get_last_sources()
        
        assert sources == []
    
    @pytest.mark.unit
    def test_reset_sources(self, tool_manager, mock_vector_store):
        """Test resetting sources from all tools"""
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = ["Test Source"]
        tool_manager.register_tool(search_tool)
        
        outline_tool = CourseOutlineTool(mock_vector_store)
        outline_tool.last_sources = ["Another Source"]
        tool_manager.register_tool(outline_tool)
        
        # Reset sources
        tool_manager.reset_sources()
        
        assert search_tool.last_sources == []
        assert outline_tool.last_sources == []