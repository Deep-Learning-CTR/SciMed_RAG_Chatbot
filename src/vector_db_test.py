"""
Interactive Vector Database Query Tool

This script allows you to test the performance of your Qdrant vector database
by querying it and viewing the top N most relevant chunks with their scores.
"""

import os
import sys
from typing import List, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.embeddings import OllamaEmbeddings


class VectorDBTester:
    """Test and query the Qdrant vector database."""
    
    def __init__(
        self,
        db_path: str = "processing/qdrant_local_db",
        collection_name: str = "paper_embeddings",
    ):
        """
        Initialize the Vector DB Tester.
        
        Args:
            db_path: Path to the Qdrant database directory
            collection_name: Name of the collection to query
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        print(f"📂 Loading Qdrant database from: {db_path}")
        self.client = QdrantClient(path=db_path)
        
        # Initialize embeddings model
        print("🤖 Initializing Ollama embeddings model...")
        self.embeddings = OllamaEmbeddings()
        
        # Verify collection exists
        self._verify_collection()
    
    def _verify_collection(self):
        """Verify that the collection exists and get its info."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"❌ Collection '{self.collection_name}' not found!")
                print(f"Available collections: {collection_names}")
                sys.exit(1)
            
            # Get collection info
            info = self.client.get_collection(self.collection_name)
            vector_count = info.points_count
            
            print(f"✅ Collection '{self.collection_name}' loaded successfully")
            print(f"📊 Total vectors in database: {vector_count}")
            print()
            
        except Exception as e:
            print(f"❌ Error loading collection: {e}")
            sys.exit(1)
    
    def query(self, query_text: str, top_k: int = 5) -> List[ScoredPoint]:
        """
        Query the vector database and return top K results.
        
        Args:
            query_text: The search query
            top_k: Number of top results to return
            
        Returns:
            List of ScoredPoint objects with results and scores
        """
        print(f"\n🔍 Searching for: '{query_text}'")
        print(f"📌 Retrieving top {top_k} results...\n")
        
        # Generate query embedding
        query_vector = self.embeddings.embed_query(query_text)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        
        return results
    
    def display_results(self, results: List[ScoredPoint], show_full_text: bool = False):
        """
        Display search results in a formatted way.
        
        Args:
            results: List of search results from Qdrant
            show_full_text: If True, show full text; if False, show preview
        """
        if not results:
            print("❌ No results found!")
            return
        
        print("=" * 100)
        print(f"{'RANK':<6} {'SCORE':<10} {'PREVIEW'}")
        print("=" * 100)
        
        for idx, result in enumerate(results, 1):
            score = result.score
            payload = result.payload
            
            # Extract text content
            text = payload.get('page_content', payload.get('text', 'N/A'))
            metadata = payload.get('metadata', {})
            
            # Display rank and score
            print(f"\n{'#' + str(idx):<6} {score:<10.4f}")
            print("-" * 100)
            
            # Display metadata if available
            if metadata:
                print("📄 Metadata:")
                for key, value in metadata.items():
                    if key not in ['page_content', 'text']:
                        print(f"   • {key}: {value}")
                print()
            
            # Display text content
            if show_full_text:
                print(f"📝 Full Text:\n{text}")
            else:
                # Show first 300 characters as preview
                preview = text[:300] + "..." if len(text) > 300 else text
                print(f"📝 Text Preview:\n{preview}")
            
            print("-" * 100)
        
        print("\n" + "=" * 100 + "\n")
    
    def interactive_mode(self):
        """Run an interactive query loop."""
        print("\n" + "=" * 100)
        print("🎯 INTERACTIVE VECTOR DATABASE QUERY MODE")
        print("=" * 100)
        print("\nCommands:")
        print("  • Enter your query to search")
        print("  • Type 'preview' to toggle preview mode (default: full text)")
        print("  • Type 'top <N>' to change number of results (e.g., 'top 10')")
        print("  • Type 'quit' or 'exit' to exit")
        print("=" * 100 + "\n")
        
        top_k = 5
        show_full_text = True
        
        while True:
            try:
                user_input = input("🔍 Enter query (or command): ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Exiting... Goodbye!")
                    break
                
                # Toggle full text display
                if user_input.lower() == 'preview':
                    show_full_text = not show_full_text
                    mode = "PREVIEW" if not show_full_text else "FULL TEXT"
                    print(f"✅ Display mode: {mode}")
                    continue
                
                # Change top K
                if user_input.lower().startswith('top '):
                    try:
                        new_k = int(user_input.split()[1])
                        if new_k > 0:
                            top_k = new_k
                            print(f"✅ Now showing top {top_k} results")
                        else:
                            print("❌ Please enter a positive number")
                    except (ValueError, IndexError):
                        print("❌ Invalid format. Use: top <number>")
                    continue
                
                # Perform search
                results = self.query(user_input, top_k=top_k)
                self.display_results(results, show_full_text=show_full_text)
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting... Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test and query the Qdrant vector database"
    )
    parser.add_argument(
        '--db-path',
        default='processing/qdrant_local_db',
        help='Path to Qdrant database (default: processing/qdrant_local_db)'
    )
    parser.add_argument(
        '--collection',
        default='paper_embeddings',
        help='Collection name (default: paper_embeddings)'
    )
    parser.add_argument(
        '--query',
        '-q',
        help='Single query to execute (interactive mode if not provided)'
    )
    parser.add_argument(
        '--top-k',
        '-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=True,
        help='Show full text instead of preview (default: True)'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = VectorDBTester(
        db_path=args.db_path,
        collection_name=args.collection
    )
    
    # Single query mode or interactive mode
    if args.query:
        results = tester.query(args.query, top_k=args.top_k)
        tester.display_results(results, show_full_text=args.full)
    else:
        tester.interactive_mode()


if __name__ == "__main__":
    main()
