from openai import OpenAI
from typing import List, Optional, Dict, Any
import json
from dataclasses import dataclass
from enum import Enum


@dataclass
class RoundResult:
    """Result from a single API round"""
    round_number: int
    tools_used: List[str]
    tool_results: Dict[str, str]
    ai_reasoning: Optional[str]
    wants_another_round: bool
    sources: List[str]
    error: Optional[str] = None


class ExecutionState(Enum):
    """State tracking for sequential execution"""
    READY = "ready"
    EXECUTING = "executing" 
    NEEDS_ANOTHER_ROUND = "needs_another_round"
    COMPLETED = "completed"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    ERROR = "error"


class SequentialToolExecutor:
    """
    State machine for managing sequential tool calls across multiple API rounds.
    Each round is a separate API call with accumulated conversation context.
    """
    
    def __init__(self, ai_generator, tool_manager, max_rounds=2):
        self.ai_generator = ai_generator
        self.tool_manager = tool_manager
        self.max_rounds = max_rounds
        self.current_round = 0
        self.accumulated_context = []  # Full conversation history
        self.round_sources = []        # Sources from all rounds
        self.execution_state = ExecutionState.READY
    
    def execute_sequential_query(self, query: str, conversation_history: Optional[str] = None) -> tuple[str, List[str]]:
        """
        Execute a query with support for up to max_rounds of tool calling.
        
        Returns:
            Tuple of (final_response, all_sources_from_all_rounds)
        """
        self.current_round = 0
        self.accumulated_context = []
        self.round_sources = []
        self.execution_state = ExecutionState.EXECUTING
        
        original_query = query
        current_context = conversation_history or ""
        
        while (self.current_round < self.max_rounds and 
               self.execution_state in [ExecutionState.EXECUTING, ExecutionState.NEEDS_ANOTHER_ROUND]):
            
            self.current_round += 1
            
            # Execute single round
            round_result = self._execute_single_round(
                query=original_query,
                round_num=self.current_round,
                accumulated_context=current_context
            )
            
            # Handle round result
            if round_result.error:
                self.execution_state = ExecutionState.ERROR
                return f"Error in round {self.current_round}: {round_result.error}", self.round_sources
                
            # Accumulate context and sources
            self.accumulated_context.append({
                'round': round_result.round_number,
                'tools_used': round_result.tools_used,
                'tool_results': round_result.tool_results,
                'intermediate_reasoning': round_result.ai_reasoning,
                'tool_decision': f"Used tools: {', '.join(round_result.tools_used) if round_result.tools_used else 'No tools'}"
            })
            self.round_sources.extend(round_result.sources)
            
            # Update context for next round
            current_context = self._build_accumulated_context(original_query)
            
            # Determine if we need another round
            if (round_result.wants_another_round and 
                self.current_round < self.max_rounds):
                self.execution_state = ExecutionState.NEEDS_ANOTHER_ROUND
            else:
                self.execution_state = ExecutionState.COMPLETED
                break
        
        # Generate final synthesis if we had multiple rounds
        if len(self.accumulated_context) > 1:
            final_response = self._synthesize_final_response(original_query, current_context)
        else:
            # Single round - use that response
            final_response = self.accumulated_context[0].get('intermediate_reasoning', 'No response generated')
        
        return final_response, self.round_sources
    
    def _build_accumulated_context(self, original_query: str) -> str:
        """
        Build complete conversation context showing the reasoning chain.
        This helps Claude understand what has already been discovered.
        """
        context_parts = [f"Original query: {original_query}"]
        
        for round_num, round_data in enumerate(self.accumulated_context, 1):
            context_parts.append(f"\n--- Round {round_num} ---")
            context_parts.append(f"AI decided to: {round_data['tool_decision']}")
            context_parts.append(f"Tool results: {round_data['tool_results']}")
            if round_data.get('intermediate_reasoning'):
                context_parts.append(f"AI reasoning: {round_data['intermediate_reasoning']}")
        
        return "\n".join(context_parts)
    
    def _generate_round_system_prompt(self, round_num: int, query: str, context: str) -> str:
        """Generate system prompt tailored to current round and context."""
        
        base_prompt = """You are an AI assistant specialized in course materials with sequential search capabilities.
    
Available Tools:
- search_course_content: Search specific course content
- get_course_outline: Get course structure and lesson lists

Sequential Search Protocol:"""
        
        if round_num == 1:
            return base_prompt + """
- This is your FIRST search opportunity
- Analyze the query to determine if multiple searches might be needed
- If the query requires information from multiple sources, plan accordingly
- Use tools strategically to gather foundational information first
- You may make another search in the next round if needed

Response Protocol:
- Provide direct answers only - no reasoning process or tool explanations
- If you need additional information to fully answer the user's question, end your response with:
  "CONTINUE_SEARCH: [brief description of what else you need]"
- If you have sufficient information, provide a complete answer without the CONTINUE_SEARCH marker
- Be brief, concise, and educational
"""
        else:  # round_num == 2
            return base_prompt + f"""
- This is your FINAL search opportunity (Round {round_num}/{self.max_rounds})
- Previous context: {context}
- Use this round to gather any additional information needed
- Synthesize all information to provide a complete answer
- This is your last chance to search - make it count

Response Protocol:
- Provide direct answers only - no reasoning process or tool explanations  
- Do NOT use "CONTINUE_SEARCH" markers - this is the final round
- Synthesize all information from both rounds into a comprehensive response
- Be brief, concise, and educational
"""
    
    def _execute_single_round(self, query: str, round_num: int, accumulated_context: str) -> RoundResult:
        """Execute a single API round with tools available."""
        
        try:
            # Generate round-specific system prompt
            system_prompt = self._generate_round_system_prompt(round_num, query, accumulated_context)
            
            # Build messages with accumulated context
            if accumulated_context:
                full_prompt = f"{query}\n\nContext from previous rounds:\n{accumulated_context}"
            else:
                full_prompt = query
            
            # Prepare API call with tools still available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            api_params = {
                "model": self.ai_generator.model,
                "temperature": 0,
                "max_tokens": 800,
                "messages": messages,
                "tools": self.tool_manager.get_tool_definitions(),  # Tools always available
                "tool_choice": "auto"
            }
            
            # Execute API call
            response = self.ai_generator.client.chat.completions.create(**api_params)
            
            # Process response
            if response.choices[0].finish_reason == "tool_calls":
                return self._handle_tool_execution_round(response, api_params, round_num)
            else:
                # No tools used - direct response
                return RoundResult(
                    round_number=round_num,
                    tools_used=[],
                    tool_results={},
                    ai_reasoning=response.choices[0].message.content,
                    wants_another_round=self._should_continue_execution(response.choices[0].message.content, []),
                    sources=[]
                )
                
        except Exception as e:
            return RoundResult(
                round_number=round_num,
                tools_used=[],
                tool_results={},
                ai_reasoning="",
                wants_another_round=False,
                sources=[],
                error=str(e)
            )
    
    def _handle_tool_execution_round(self, initial_response, base_params: Dict[str, Any], round_num: int) -> RoundResult:
        """Handle tool execution for a single round while preserving full context."""
        
        messages = base_params["messages"].copy()
        tools_used = []
        tool_results = {}
        sources = []
        
        # Add AI's tool use response
        messages.append({
            "role": "assistant",
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls
        })
        
        # Execute all tool calls
        for tool_call in initial_response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                tool_name = tool_call.function.name
                tools_used.append(tool_name)
                
                # Execute tool
                tool_result = self.tool_manager.execute_tool(
                    tool_name,
                    **json.loads(tool_call.function.arguments)
                )
                tool_results[tool_name] = tool_result
                
                # Collect sources
                tool_sources = self.tool_manager.get_last_sources()
                sources.extend(tool_sources)
                self.tool_manager.reset_sources()
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
        
        # Get AI's response to tool results with continuation analysis prompt
        continuation_system = f"""Based on the tool results from round {round_num}, provide your response.

If you need additional information to fully answer the user's question, end your response with:
"CONTINUE_SEARCH: [brief description of what else you need]"

If you have sufficient information, provide a complete answer without the CONTINUE_SEARCH marker.

Response Protocol:
- Provide direct answers only - no reasoning process or tool explanations
- Be brief, concise, and educational
"""

        final_messages = [
            {"role": "system", "content": continuation_system}
        ] + messages
        
        final_params = {
            **base_params,
            "messages": final_messages,
        }
        final_params.pop("tools", None)
        final_params.pop("tool_choice", None)
        
        final_response = self.ai_generator.client.chat.completions.create(**final_params)
        ai_reasoning = final_response.choices[0].message.content
        
        # Check if AI wants to continue
        wants_another_round = "CONTINUE_SEARCH:" in ai_reasoning
        
        return RoundResult(
            round_number=round_num,
            tools_used=tools_used,
            tool_results=tool_results,
            ai_reasoning=ai_reasoning,
            wants_another_round=wants_another_round,
            sources=sources
        )
    
    def _should_continue_execution(self, response_content: str, tools_used: List[str]) -> bool:
        """
        Determine if AI needs another round based on response analysis.
        Uses multiple heuristics rather than relying solely on AI's explicit request.
        """
        
        # Explicit continuation signals
        continuation_phrases = [
            "need to search for more",
            "let me search for additional", 
            "I should also check",
            "need another search",
            "incomplete information",
            "CONTINUE_SEARCH:"
        ]
        
        # Check for explicit signals
        response_lower = response_content.lower()
        has_continuation_signal = any(phrase.lower() in response_lower for phrase in continuation_phrases)
        
        # Check if no tools were used (might indicate planning phase)
        no_tools_used = len(tools_used) == 0
        
        return has_continuation_signal or no_tools_used
    
    def _synthesize_final_response(self, original_query: str, full_context: str) -> str:
        """Generate final synthesized response from all rounds."""
        
        synthesis_prompt = f"""You have completed multiple search rounds to answer this query: "{original_query}"

Here is everything you discovered:
{full_context}

Now provide a comprehensive final answer that synthesizes all the information gathered across all rounds. 
Remove any "CONTINUE_SEARCH" markers and provide a polished, complete response.

Response Protocol:
- Provide direct answers only - no reasoning process explanations
- Be brief, concise, and educational
- Synthesize information from all rounds into a cohesive answer
"""

        messages = [
            {"role": "system", "content": "You are synthesizing information from multiple search rounds to provide a comprehensive final answer."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        api_params = {
            "model": self.ai_generator.model,
            "temperature": 0,
            "max_tokens": 1000,  # Slightly higher for synthesis
            "messages": messages
        }
        
        response = self.ai_generator.client.chat.completions.create(**api_params)
        return response.choices[0].message.content


class AIGenerator:
    """Handles interactions with DeepSeek's API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
- **search_course_content**: For questions about specific course content or detailed educational materials
- **get_course_outline**: For questions about course structure, lesson lists, or course overviews

Tool Usage Guidelines:
- **Maximum 2 tool call rounds per query**
- Use initial search to gather foundational information
- If initial results suggest additional searches would be helpful, use a second round
- For outline/structure questions: Use get_course_outline to return course title, course link, and complete lesson list
- For content questions: Use search_course_content for specific material within courses
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool first, then provide the complete outline information including course title, link, and all lesson numbers with titles
- **Course content questions**: Use search_course_content tool first, then answer based on results
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
5. **Complete for outlines** - Include course title, course link, and full lesson list with numbers and titles
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         enable_sequential: bool = True) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            enable_sequential: Whether to enable sequential tool calling (default: True)
            
        Returns:
            Generated response as string
        """
        
        # Use sequential execution if tools and tool_manager are available and sequential is enabled
        if tools and tool_manager and enable_sequential:
            executor = SequentialToolExecutor(self, tool_manager, max_rounds=2)
            final_response, sources = executor.execute_sequential_query(query, conversation_history)
            
            # Store sources in tool_manager for compatibility with existing code
            # We need to set the sources back so RAGSystem can retrieve them
            if hasattr(tool_manager, 'set_sequential_sources'):
                tool_manager.set_sequential_sources(sources)
            else:
                # Fallback: set sources on first available tool
                for tool in tool_manager.tools.values():
                    if hasattr(tool, 'last_sources'):
                        tool.last_sources = sources
                        break
            
            return final_response
        
        # Fallback to original single-round execution for backward compatibility
        return self._generate_single_round_response(query, conversation_history, tools, tool_manager)
    
    def _generate_single_round_response(self, query: str,
                                      conversation_history: Optional[str] = None,
                                      tools: Optional[List] = None,
                                      tool_manager=None) -> str:
        """
        Original single-round response generation for backward compatibility.
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare messages for OpenAI format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        # Get response from DeepSeek
        response = self.client.chat.completions.create(**api_params)
        
        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({
            "role": "assistant", 
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls
        })
        
        # Execute all tool calls and collect results
        for tool_call in initial_response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name,
                    **json.loads(tool_call.function.arguments)
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content