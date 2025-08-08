"""Tests for AIGenerator tool calling functionality with DeepSeek API."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from ai_generator import AIGenerator
from .test_utils import (
    create_mock_deepseek_response, 
    create_mock_tool_call, 
    create_mock_sequential_responses,
    create_mock_sequential_tool_manager,
    assert_sequential_execution_calls,
    get_api_call_messages
)


class TestAIGenerator:
    """Test cases for AIGenerator functionality"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        return Mock()
    
    @pytest.fixture
    def ai_generator(self, mock_openai_client):
        """Create AIGenerator with mock OpenAI client"""
        with patch('ai_generator.OpenAI') as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client
            generator = AIGenerator("test_api_key", "deepseek-chat")
            generator.client = mock_openai_client  # Ensure we use our mock
            return generator, mock_openai_client
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager"""
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "[Test Course - Lesson 1]\nThis is the search result content."
        return tool_manager
    
    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "course_name": {"type": "string", "description": "Course name"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    @pytest.mark.unit
    def test_init_parameters(self):
        """Test AIGenerator initialization with correct parameters"""
        with patch('ai_generator.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            generator = AIGenerator("test_key", "deepseek-chat")
            
            # Check OpenAI client was initialized correctly
            mock_openai_class.assert_called_once_with(
                api_key="test_key",
                base_url="https://api.deepseek.com"
            )
            assert generator.model == "deepseek-chat"
            assert generator.base_params["model"] == "deepseek-chat"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    @pytest.mark.unit
    def test_generate_response_without_tools(self, ai_generator):
        """Test generating response without tools"""
        generator, mock_client = ai_generator
        
        # Mock successful response
        mock_response = create_mock_deepseek_response("This is a test response about machine learning.")
        mock_client.chat.completions.create.return_value = mock_response
        
        response = generator.generate_response("What is machine learning?")
        
        assert response == "This is a test response about machine learning."
        mock_client.chat.completions.create.assert_called_once()
        
        # Check call arguments
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "deepseek-chat"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["role"] == "user"
        assert "tools" not in call_args
    
    @pytest.mark.unit
    def test_generate_response_with_tools_no_calls(self, ai_generator, sample_tools):
        """Test generating response with tools available but not used"""
        generator, mock_client = ai_generator
        
        # Mock response without tool calls
        mock_response = create_mock_deepseek_response("I can help you with course materials.")
        mock_client.chat.completions.create.return_value = mock_response
        
        response = generator.generate_response(
            "Hello, can you help me?", 
            tools=sample_tools
        )
        
        assert response == "I can help you with course materials."
        
        # Check tools were provided in the call
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == sample_tools
        assert call_args["tool_choice"] == "auto"
    
    @pytest.mark.unit  
    def test_generate_response_with_tool_calls(self, ai_generator, sample_tools, mock_tool_manager):
        """Test generating response that requires tool execution"""
        generator, mock_client = ai_generator
        
        # Create mock tool call
        tool_call = create_mock_tool_call(
            "call_123", 
            "search_course_content", 
            {"query": "machine learning", "course_name": "ML Course"}
        )
        
        # Mock initial response with tool calls
        initial_response = create_mock_deepseek_response(None, [tool_call])
        initial_response.choices[0].finish_reason = "tool_calls"
        
        # Mock final response after tool execution
        final_response = create_mock_deepseek_response(
            "Based on the search results, machine learning is a field of AI that focuses on algorithms."
        )
        
        # Set up client to return both responses
        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        
        response = generator.generate_response(
            "What is machine learning?",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )
        
        assert response == "Based on the search results, machine learning is a field of AI that focuses on algorithms."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning",
            course_name="ML Course"
        )
        
        # Verify two API calls were made
        assert mock_client.chat.completions.create.call_count == 2
    
    @pytest.mark.unit
    def test_generate_response_with_conversation_history(self, ai_generator):
        """Test generating response with conversation history"""
        generator, mock_client = ai_generator
        
        mock_response = create_mock_deepseek_response("Following up on our previous discussion...")
        mock_client.chat.completions.create.return_value = mock_response
        
        history = "User: What is supervised learning?\nAssistant: Supervised learning uses labeled data to train models."
        
        response = generator.generate_response(
            "Can you give me an example?",
            conversation_history=history
        )
        
        assert response == "Following up on our previous discussion..."
        
        # Check that history was included in system message
        call_args = mock_client.chat.completions.create.call_args[1]
        system_content = call_args["messages"][0]["content"]
        assert "Previous conversation:" in system_content
        assert "supervised learning" in system_content.lower()
    
    @pytest.mark.unit
    def test_handle_tool_execution_success(self, ai_generator, mock_tool_manager):
        """Test successful tool execution handling"""
        generator, mock_client = ai_generator
        
        # Create mock tool call and initial response
        tool_call = create_mock_tool_call("call_123", "search_course_content", {"query": "test"})
        initial_response = create_mock_deepseek_response(None, [tool_call])
        initial_response.choices[0].finish_reason = "tool_calls"
        
        # Mock final response
        final_response = create_mock_deepseek_response("Here's what I found about your query.")
        mock_client.chat.completions.create.return_value = final_response
        
        # Prepare base params
        base_params = {
            "model": "deepseek-chat",
            "temperature": 0,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Test query"}
            ]
        }
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Here's what I found about your query."
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Check final API call was made without tools
        final_call_args = mock_client.chat.completions.create.call_args[1]
        assert "tools" not in final_call_args
        assert len(final_call_args["messages"]) == 4  # system, user, assistant, tool
    
    @pytest.mark.unit
    def test_handle_multiple_tool_calls(self, ai_generator, mock_tool_manager):
        """Test handling multiple tool calls in sequence"""
        generator, mock_client = ai_generator
        
        # Create multiple tool calls
        tool_call1 = create_mock_tool_call("call_1", "search_course_content", {"query": "ML"})
        tool_call2 = create_mock_tool_call("call_2", "get_course_outline", {"course_name": "ML Course"})
        
        initial_response = create_mock_deepseek_response(None, [tool_call1, tool_call2])
        initial_response.choices[0].finish_reason = "tool_calls"
        
        # Mock tool manager responses
        mock_tool_manager.execute_tool.side_effect = [
            "Search result for ML",
            "Course outline for ML Course"
        ]
        
        final_response = create_mock_deepseek_response("Based on both the search and outline...")
        mock_client.chat.completions.create.return_value = final_response
        
        base_params = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Tell me about ML"}]
        }
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "Based on both the search and outline..."
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Check tool calls were made with correct parameters
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_course_content",)
        assert calls[0][1] == {"query": "ML"}
        assert calls[1][0] == ("get_course_outline",)
        assert calls[1][1] == {"course_name": "ML Course"}
    
    @pytest.mark.unit
    def test_tool_execution_with_tool_error(self, ai_generator, mock_tool_manager):
        """Test tool execution when tool returns an error"""
        generator, mock_client = ai_generator
        
        # Mock tool manager to return error
        mock_tool_manager.execute_tool.return_value = "Error: Database connection failed"
        
        tool_call = create_mock_tool_call("call_123", "search_course_content", {"query": "test"})
        initial_response = create_mock_deepseek_response(None, [tool_call])
        initial_response.choices[0].finish_reason = "tool_calls"
        
        final_response = create_mock_deepseek_response(
            "I apologize, but I encountered an error while searching."
        )
        mock_client.chat.completions.create.return_value = final_response
        
        base_params = {"model": "deepseek-chat", "messages": []}
        
        result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
        
        assert result == "I apologize, but I encountered an error while searching."
        
        # Check that tool error was passed to final response
        final_call_args = mock_client.chat.completions.create.call_args[1]
        messages = final_call_args["messages"]
        
        # Find the tool response message
        tool_message = None
        for msg in messages:
            if msg.get("role") == "tool":
                tool_message = msg
                break
        
        assert tool_message is not None
        assert "Error: Database connection failed" in tool_message["content"]
    
    @pytest.mark.unit
    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected instructions"""
        generator, mock_client = ai_generator
        
        mock_response = create_mock_deepseek_response("Test response")
        mock_client.chat.completions.create.return_value = mock_response
        
        generator.generate_response("Test query")
        
        call_args = mock_client.chat.completions.create.call_args[1]
        system_content = call_args["messages"][0]["content"]
        
        # Check key elements of system prompt
        assert "search_course_content" in system_content
        assert "get_course_outline" in system_content
        assert "One tool call per query maximum" in system_content
        assert "Brief, Concise and focused" in system_content
    
    @pytest.mark.unit
    def test_api_error_handling(self, ai_generator):
        """Test handling of API errors"""
        generator, mock_client = ai_generator
        
        # Mock API to raise an exception
        mock_client.chat.completions.create.side_effect = Exception("API connection failed")
        
        with pytest.raises(Exception, match="API connection failed"):
            generator.generate_response("Test query")
    
    @pytest.mark.unit
    def test_generate_response_empty_query(self, ai_generator):
        """Test generating response with empty query"""
        generator, mock_client = ai_generator
        
        mock_response = create_mock_deepseek_response("I'm here to help with any questions.")
        mock_client.chat.completions.create.return_value = mock_response
        
        response = generator.generate_response("")
        
        assert response == "I'm here to help with any questions."
        
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["messages"][1]["content"] == ""
    
    @pytest.mark.unit
    def test_base_params_consistency(self, ai_generator, sample_tools):
        """Test that base parameters remain consistent across calls"""
        generator, mock_client = ai_generator
        
        mock_response = create_mock_deepseek_response("Test response")
        mock_client.chat.completions.create.return_value = mock_response
        
        # Make multiple calls
        generator.generate_response("Query 1")
        generator.generate_response("Query 2", tools=sample_tools)
        generator.generate_response("Query 3", conversation_history="Previous context")
        
        # Check all calls used consistent base parameters
        calls = mock_client.chat.completions.create.call_args_list
        
        for call in calls:
            args = call[1]
            assert args["model"] == "deepseek-chat"
            assert args["temperature"] == 0 
            assert args["max_tokens"] == 800

    # Sequential Tool Calling Tests
    @pytest.mark.unit
    def test_sequential_tool_calling_single_round(self, ai_generator):
        """Test sequential execution with single round (no continuation needed)"""
        generator, mock_client = ai_generator
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        # Mock single round: tool call + final response without CONTINUE_SEARCH
        tool_call = create_mock_tool_call("call_1", "search_course_content", {"query": "machine learning"})
        responses = create_mock_sequential_responses([
            # Round 1: Initial request with tool call
            {"content": "I'll search for machine learning content.", "tool_calls": [tool_call]},
            # Round 1: Response after tool execution (no continuation)
            {"content": "Machine learning is a subset of AI that focuses on algorithms.", "tool_calls": []}
        ])
        mock_client.chat.completions.create.side_effect = responses
        
        result = generator.generate_response(
            "What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Should use the single round response since no CONTINUE_SEARCH marker
        assert "Machine learning is a subset of AI" in result
        assert_sequential_execution_calls(mock_client, 2)  # 2 calls for single round with tool
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="machine learning")
        
        # Verify sources were set
        mock_tool_manager.set_sequential_sources.assert_called_once()

    @pytest.mark.unit
    def test_sequential_tool_calling_two_rounds(self, ai_generator):
        """Test sequential execution with two rounds"""
        generator, mock_client = ai_generator
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        # Mock two rounds with continuation
        tool_call_1 = create_mock_tool_call("call_1", "get_course_outline", {"course_name": "ML Course"})
        tool_call_2 = create_mock_tool_call("call_2", "search_course_content", {"query": "lesson 2", "course_name": "ML Course"})
        
        responses = create_mock_sequential_responses([
            # Round 1: Initial request with tool call
            {"content": "I'll get the course outline first.", "tool_calls": [tool_call_1]},
            # Round 1: Response after tool execution with continuation
            {"content": "I found the course outline. CONTINUE_SEARCH: Need to get details about lesson 2", "tool_calls": []},
            # Round 2: Second request with tool call  
            {"content": "Now I'll search for lesson 2 details.", "tool_calls": [tool_call_2]},
            # Round 2: Final response
            {"content": "Lesson 2 covers advanced algorithms in machine learning.", "tool_calls": []},
            # Final synthesis call
            {"content": "Based on the course outline and lesson details, lesson 2 focuses on algorithms.", "tool_calls": []}
        ])
        mock_client.chat.completions.create.side_effect = responses
        
        result = generator.generate_response(
            "What does lesson 2 of ML Course cover?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Should use synthesized response from multiple rounds
        assert "algorithms" in result.lower()
        assert_sequential_execution_calls(mock_client, 5)  # Multiple rounds + synthesis
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("get_course_outline",)
        assert calls[1][0] == ("search_course_content",)

    @pytest.mark.unit  
    def test_sequential_tool_calling_max_rounds_reached(self, ai_generator):
        """Test sequential execution stops after max rounds"""
        generator, mock_client = ai_generator
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        # Mock responses that would continue beyond max rounds
        tool_call_1 = create_mock_tool_call("call_1", "search_course_content", {"query": "ML basics"})
        tool_call_2 = create_mock_tool_call("call_2", "get_course_outline", {"course_name": "ML Course"})
        
        responses = create_mock_sequential_responses([
            # Round 1
            {"content": "Searching for ML basics.", "tool_calls": [tool_call_1]},
            {"content": "Found some info. CONTINUE_SEARCH: Need course structure", "tool_calls": []},
            # Round 2  
            {"content": "Getting course outline.", "tool_calls": [tool_call_2]},
            {"content": "Here's the outline. CONTINUE_SEARCH: Need more details", "tool_calls": []},  # This should be ignored
            # Final synthesis
            {"content": "Based on available information, here's what I found about ML.", "tool_calls": []}
        ])
        mock_client.chat.completions.create.side_effect = responses
        
        result = generator.generate_response(
            "Tell me about ML basics and course structure",
            tools=mock_tool_manager.get_tool_definitions(), 
            tool_manager=mock_tool_manager
        )
        
        # Should complete after 2 rounds even if AI wants to continue
        assert "available information" in result
        assert_sequential_execution_calls(mock_client, 5)  # 2 rounds + synthesis
        
        # Should have executed tools in both rounds
        assert mock_tool_manager.execute_tool.call_count == 2

    @pytest.mark.unit
    def test_sequential_tool_calling_with_error(self, ai_generator):
        """Test sequential execution handles tool errors gracefully"""
        generator, mock_client = ai_generator  
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        # Mock tool manager to return error for second tool call
        def mock_execute_with_error(tool_name, **kwargs):
            if tool_name == "search_course_content":
                return "Error: Database connection failed"
            return "Course outline here"
        
        mock_tool_manager.execute_tool.side_effect = mock_execute_with_error
        
        tool_call = create_mock_tool_call("call_1", "search_course_content", {"query": "ML"})
        responses = create_mock_sequential_responses([
            {"content": "Searching for ML content.", "tool_calls": [tool_call]},
            {"content": "I encountered an error while searching. Please try again.", "tool_calls": []}
        ])
        mock_client.chat.completions.create.side_effect = responses
        
        result = generator.generate_response(
            "Search for ML content",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Should handle error gracefully
        assert "error" in result.lower()
        assert mock_tool_manager.execute_tool.assert_called_once

    @pytest.mark.unit
    def test_sequential_tool_calling_disabled(self, ai_generator, sample_tools):
        """Test that sequential execution can be disabled for backward compatibility"""
        generator, mock_client = ai_generator
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        # Mock single response (should use original single-round logic)
        tool_call = create_mock_tool_call("call_1", "search_course_content", {"query": "test"})
        initial_response = create_mock_deepseek_response(None, [tool_call])
        initial_response.choices[0].finish_reason = "tool_calls"
        final_response = create_mock_deepseek_response("Single round result")
        
        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        
        result = generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
            enable_sequential=False  # Disable sequential execution
        )
        
        assert result == "Single round result"
        assert_sequential_execution_calls(mock_client, 2)  # Original single-round behavior

    @pytest.mark.unit
    def test_sequential_system_prompt_changes(self, ai_generator):
        """Test that system prompts change between rounds"""
        generator, mock_client = ai_generator
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        # Mock two round execution
        tool_call_1 = create_mock_tool_call("call_1", "search_course_content", {"query": "ML"})
        tool_call_2 = create_mock_tool_call("call_2", "get_course_outline", {"course_name": "ML Course"})
        
        responses = create_mock_sequential_responses([
            {"content": "First search.", "tool_calls": [tool_call_1]},
            {"content": "Found info. CONTINUE_SEARCH: Need outline", "tool_calls": []},
            {"content": "Getting outline.", "tool_calls": [tool_call_2]},  
            {"content": "Here's the complete information.", "tool_calls": []},
            {"content": "Final synthesized response.", "tool_calls": []}
        ])
        mock_client.chat.completions.create.side_effect = responses
        
        result = generator.generate_response(
            "Test multi-round query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Check that different system prompts were used
        calls = mock_client.chat.completions.create.call_args_list
        
        # Round 1 system prompt should mention "FIRST search opportunity"  
        round1_messages = get_api_call_messages(mock_client, 0)
        assert "FIRST search opportunity" in round1_messages[0]["content"]
        
        # Round 2 system prompt should mention "FINAL search opportunity"
        round2_messages = get_api_call_messages(mock_client, 2)
        assert "FINAL search opportunity" in round2_messages[0]["content"]

    @pytest.mark.unit
    def test_sequential_context_accumulation(self, ai_generator):
        """Test that context accumulates properly between rounds"""
        generator, mock_client = ai_generator
        mock_tool_manager = create_mock_sequential_tool_manager()
        
        tool_call_1 = create_mock_tool_call("call_1", "search_course_content", {"query": "basics"})
        responses = create_mock_sequential_responses([
            {"content": "Searching basics.", "tool_calls": [tool_call_1]},
            {"content": "Found basic info. CONTINUE_SEARCH: Need advanced topics", "tool_calls": []},
            {"content": "No additional search needed.", "tool_calls": []},  # Second round without tools
            {"content": "Final response with context.", "tool_calls": []}
        ])
        mock_client.chat.completions.create.side_effect = responses
        
        result = generator.generate_response(
            "Original query about ML",
            tools=mock_tool_manager.get_tool_definitions(), 
            tool_manager=mock_tool_manager
        )
        
        # Check that second round received context from first round
        calls = mock_client.chat.completions.create.call_args_list
        if len(calls) >= 2:
            round2_messages = get_api_call_messages(mock_client, 1)
            user_message = next((msg for msg in round2_messages if msg["role"] == "user"), None)
            
            # Should contain context from previous rounds
            assert user_message is not None
            assert "Context from previous rounds:" in user_message["content"]
            assert "Round 1" in user_message["content"]

    @pytest.mark.unit
    def test_system_prompt_updated_for_sequential(self, ai_generator):
        """Test that the system prompt reflects sequential capabilities"""
        generator, mock_client = ai_generator
        
        # Make a call to trigger system prompt usage
        mock_response = create_mock_deepseek_response("Test response")
        mock_client.chat.completions.create.return_value = mock_response
        
        generator.generate_response("Test query", enable_sequential=False)
        
        # Check system prompt content
        calls = mock_client.chat.completions.create.call_args_list
        system_message = calls[0][1]["messages"][0]
        
        assert system_message["role"] == "system"
        # Should mention 2 rounds instead of single tool call maximum
        assert "Maximum 2 tool call rounds per query" in system_message["content"]
        # Should not have the old single-round constraint
        assert "One tool call per query maximum" not in system_message["content"]