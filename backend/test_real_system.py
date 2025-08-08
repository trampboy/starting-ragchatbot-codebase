#!/usr/bin/env python3
"""
Real system test to verify the RAG system actually works with course data.
This will help identify if the "query failed" issue is in the actual implementation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from rag_system import RAGSystem


def test_real_rag_system():
    """Test the actual RAG system with real course data"""
    print("🔍 Testing RAG System with Real Course Data")
    print("=" * 50)
    
    try:
        # Initialize the RAG system 
        print("1. Initializing RAG system...")
        rag = RAGSystem(config)
        print("✅ RAG system initialized successfully")
        
        # Check if we have course data
        print("\n2. Checking course data...")
        analytics = rag.get_course_analytics()
        print(f"   Total courses: {analytics['total_courses']}")
        print(f"   Course titles: {analytics['course_titles']}")
        
        if analytics['total_courses'] == 0:
            print("❌ No courses found - loading from docs folder...")
            docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
            if os.path.exists(docs_path):
                courses, chunks = rag.add_course_folder(docs_path, clear_existing=False)
                print(f"✅ Loaded {courses} courses with {chunks} chunks")
            else:
                print(f"❌ Docs folder not found: {docs_path}")
                return
        
        # Test various types of queries
        test_queries = [
            "What is machine learning?",
            "Tell me about computer use",
            "Explain neural networks",
            "What are the course topics?",
            "How does supervised learning work?"
        ]
        
        print(f"\n3. Testing {len(test_queries)} queries...")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔸 Query {i}: {query}")
            
            try:
                response, sources = rag.query(query)
                
                print(f"   Response: {response[:200]}{'...' if len(response) > 200 else ''}")
                print(f"   Sources: {len(sources)} found")
                if sources:
                    for j, source in enumerate(sources[:2], 1):  # Show first 2 sources
                        print(f"     {j}. {source}")
                
                # Check if response indicates failure
                if "query failed" in response.lower() or "error" in response.lower():
                    print("   ❌ Query appears to have failed")
                else:
                    print("   ✅ Query completed successfully")
                    
            except Exception as e:
                print(f"   ❌ Query failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    # Set up environment variable if not already set
    if not config.DEEPSEEK_API_KEY:
        print("❌ DEEPSEEK_API_KEY not set. Please set it in your .env file.")
        print("   Example: DEEPSEEK_API_KEY=your_key_here")
        sys.exit(1)
    
    test_real_rag_system()