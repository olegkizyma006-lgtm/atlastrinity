"""
AtlasTrinity Long-Term Memory

ChromaDB-based vector memory for storing:
- Lessons learned from errors
- Successful strategies
- Task patterns

This enables the system to learn from past experience.
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .config import CONFIG_ROOT, MEMORY_DIR
from .config_loader import config
from .logger import logger

# ChromaDB storage path - use config if available, else default
CHROMA_DIR = config.get("mcp.memory.chroma_path")
if not CHROMA_DIR:
    # Handle dynamic variable if it was just loaded raw (rare, usually resolved by loader)
    CHROMA_DIR = str(MEMORY_DIR / "chroma")
else:
    # Ensure variables are expanded if config loader didn't (loader usually does)
    CHROMA_DIR = os.path.expandvars(CHROMA_DIR)



class LongTermMemory:
    """
    Manages long-term vector memory using ChromaDB.

    Collections:
    - lessons: Error patterns and their solutions
    - strategies: Successful task strategies
    - context: Task context and outcomes
    """

    def __init__(self):
        if not CHROMADB_AVAILABLE:
            logger.warning("[MEMORY] ChromaDB not installed. Running without long-term memory.")
            self.available = False
            return

        os.makedirs(CHROMA_DIR, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)

            # Initialize collections
            self.lessons = self.client.get_or_create_collection(
                name="lessons", metadata={"description": "Error patterns and solutions"}
            )

            self.strategies = self.client.get_or_create_collection(
                name="strategies",
                metadata={"description": "Successful task execution strategies"},
            )

            self.knowledge = self.client.get_or_create_collection(
                name="knowledge_graph_nodes",
                metadata={"description": "Semantic embedding of Knowledge Graph nodes"},
            )

            self.conversations = self.client.get_or_create_collection(
                name="conversations",
                metadata={"description": "Summaries of past chat sessions for semantic recall"},
            )

            self.behavior_deviations = self.client.get_or_create_collection(
                name="behavior_deviations",
                metadata={"description": "Successful logic deviations from original plans"},
            )

            self.available = True
            logger.info(f"[MEMORY] ChromaDB initialized at {CHROMA_DIR}")
            logger.info(
                f"[MEMORY] Lessons: {self.lessons.count()} | Strategies: {self.strategies.count()} | Conversations: {self.conversations.count()}"
            )

        except Exception as e:
            logger.error(f"[MEMORY] Failed to initialize ChromaDB: {e}")
            self.available = False

    def remember_error(
        self,
        error: str,
        solution: str,
        context: Dict[str, Any],
        task_description: str = "",
    ) -> bool:
        """
        Store an error pattern with its solution.

        Args:
            error: The error message or description
            solution: How the error was resolved
            context: Additional context (step, tool, path, etc.)
            task_description: What task was being attempted
        """
        if not self.available:
            return False

        try:
            doc_id = f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(error) % 10000}"

            # Create document text for embedding
            document = f"Error: {error}\nSolution: {solution}\nTask: {task_description}"

            # Metadata for filtering
            metadata = {
                "error_type": (type(error).__name__ if hasattr(error, "__name__") else "string"),
                "timestamp": datetime.now().isoformat(),
                "tool": context.get("tool", ""),
                "step_id": str(context.get("step_id", "")),
                "success": context.get("success", False),
            }

            self.lessons.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])

            logger.info(f"[MEMORY] Stored lesson: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"[MEMORY] Failed to store lesson: {e}")
            return False

    def remember_strategy(
        self, task: str, plan_steps: List[str], outcome: str, success: bool
    ) -> bool:
        """
        Store a task execution strategy.

        Args:
            task: Task description
            plan_steps: List of steps taken
            outcome: Final result
            success: Whether the strategy worked
        """
        if not self.available:
            return False

        try:
            doc_id = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(task) % 10000}"

            # Create document text
            steps_text = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(plan_steps)])
            document = f"Task: {task}\n\nSteps:\n{steps_text}\n\nOutcome: {outcome}"

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "steps_count": len(plan_steps),
            }

            self.strategies.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])

            logger.info(f"[MEMORY] Stored strategy: {doc_id} (success={success})")
            return True

        except Exception as e:
            logger.error(f"[MEMORY] Failed to store strategy: {e}")
            return False

    def recall_similar_errors(self, error: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar past errors and their solutions.

        Args:
            error: Current error to find similar cases for
            n_results: Number of results to return

        Returns:
            List of dicts with {document, metadata, distance}
        """
        if not self.available or self.lessons.count() == 0:
            return []

        try:
            results = self.lessons.query(
                query_texts=[error],
                n_results=min(n_results, self.lessons.count()),
                include=["documents", "metadatas", "distances"],
            )

            similar = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    similar.append(
                        {
                            "document": doc,
                            "metadata": (
                                results["metadatas"][0][i] if results["metadatas"] else {}
                            ),
                            "distance": (
                                results["distances"][0][i] if results["distances"] else 1.0
                            ),
                        }
                    )

            logger.info(f"[MEMORY] Found {len(similar)} similar errors")
            return similar

        except Exception as e:
            logger.error(f"[MEMORY] Failed to recall errors: {e}")
            return []

    def recall_similar_tasks(
        self, task: str, n_results: int = 3, only_successful: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find similar past tasks and their strategies.

        Args:
            task: Current task description
            n_results: Number of results to return
            only_successful: Only return successful strategies

        Returns:
            List of dicts with {document, metadata, distance}
        """
        if not self.available or self.strategies.count() == 0:
            return []

        try:
            where_filter = {"success": True} if only_successful else None

            results = self.strategies.query(
                query_texts=[task],
                n_results=min(n_results, self.strategies.count()),
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )

            similar = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    similar.append(
                        {
                            "document": doc,
                            "metadata": (
                                results["metadatas"][0][i] if results["metadatas"] else {}
                            ),
                            "distance": (
                                results["distances"][0][i] if results["distances"] else 1.0
                            ),
                        }
                    )

            logger.info(f"[MEMORY] Found {len(similar)} similar tasks")
            return similar

        except Exception as e:
            # Catch specific ChromaDB internal errors that might occur during query execution
            if "Internal error" in str(e) or "Error finding id" in str(e):
                 logger.warning(f"[MEMORY] ChromaDB internal query error (ignoring): {e}")
                 return []
            logger.error(f"[MEMORY] Failed to recall tasks: {e}")
            return []

    def add_knowledge_node(self, node_id: str, text: str, metadata: Dict[str, Any], namespace: str = "global", task_id: str = "") -> bool:
        """Add a knowledge graph node to vector store."""
        if not self.available:
            return False

        try:
            self.knowledge.upsert(ids=[node_id], documents=[text], metadatas=[metadata])
            logger.info(f"[MEMORY] Added knowledge node: {node_id}")
            return True
        except Exception as e:
            logger.error(f"[MEMORY] Failed to add knowledge node: {e}")
            return False

    def remember_conversation(self, session_id: str, summary: str, metadata: Dict[str, Any] = None) -> bool:
        """Store a conversation summary in vector memory."""
        if not self.available:
            return False
            
        try:
            doc_id = f"conv_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.conversations.upsert(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }]
            )
            logger.info(f"[MEMORY] Stored conversation summary: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"[MEMORY] Failed to store conversation: {e}")
            return False

    def recall_similar_conversations(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Find past conversations related to the current query."""
        if not self.available or self.conversations.count() == 0:
            return []
            
        try:
            results = self.conversations.query(
                query_texts=[query],
                n_results=min(n_results, self.conversations.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            similar = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    similar.append({
                        "summary": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 1.0
                    })
            return similar
        except Exception as e:
            logger.error(f"[MEMORY] Failed to recall conversations: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.available:
            return {"available": False}

        return {
            "available": True,
            "lessons_count": self.lessons.count(),
            "strategies_count": self.strategies.count(),
            "conversations_count": self.conversations.count(),
            "path": CHROMA_DIR,
        }

    def consolidate(self, logs: List[Dict[str, Any]], llm_summarizer=None) -> int:
        """
        Consolidate logs into lessons (for nightly processing).

        Args:
            logs: List of log entries with error/success data
            llm_summarizer: Optional LLM for generating summaries

        Returns:
            Number of new lessons created
        """
        if not self.available:
            return 0

        new_lessons = 0

        # Group errors by similarity
        errors = [log for log in logs if log.get("type") == "error" or log.get("success") is False]

        for error_log in errors:
            # Check if we already have this lesson
            existing = self.recall_similar_errors(
                error_log.get("error", str(error_log)), n_results=1
            )

            # Only add if not too similar
            if not existing or existing[0].get("distance", 1.0) > 0.1:
                self.remember_error(
                    error=error_log.get("error", str(error_log)),
                    solution=error_log.get("solution", "Unknown"),
                    context=error_log,
                    task_description=error_log.get("task", ""),
                )
                new_lessons += 1

        logger.info(f"[MEMORY] Consolidated {new_lessons} new lessons from {len(logs)} logs")
        return new_lessons

    def remember_behavioral_change(
        self, 
        original_intent: str, 
        deviation: str, 
        reason: str, 
        result: str, 
        context: Dict[str, Any],
        decision_factors: Dict[str, Any] = None
    ) -> bool:
        """
        Store a successful logic deviation with decision context.
        
        Args:
            original_intent: What was originally planned
            deviation: What was actually done
            reason: Why the change was made
            result: The outcome
            context: Execution context (step_id, etc.)
            decision_factors: Key environmental factors driving the decision (e.g. "time_pressure", "resource_unavailability")
        """
        if not self.available: return False
        try:
            doc_id = f"deviation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(original_intent) % 10000}"
            
            # Format factors for semantic understanding
            factors_text = ""
            if decision_factors:
                factors_text = "\nDecision Factors:\n" + "\n".join([f"- {k}: {v}" for k, v in decision_factors.items()])
            
            document = (
                f"Original Intent: {original_intent}\n"
                f"Deviated To: {deviation}\n"
                f"Reason: {reason}\n"
                f"Outcome: {result}"
                f"{factors_text}"
            )
            
            # Enrich metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "step_id": str(context.get("step_id", "")),
                "success": True
            }
            # Flatten simple factors into metadata for filtering
            if decision_factors:
                for k, v in decision_factors.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[f"factor_{k}"] = v
            
            self.behavior_deviations.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"[MEMORY] Stored behavior deviation in ChromaDB: {doc_id}")
            
            # 2. Sync to Relational DB (SQL) for auditing
            try:
                from .db.manager import db_manager
                from .db.schema import BehavioralDeviation
                
                async def _sync_to_sql():
                    async with await db_manager.get_session() as session:
                        deviation_entry = BehavioralDeviation(
                            session_id=context.get("db_session_id") or context.get("session_id"),
                            step_id=context.get("step_id"),
                            original_intent=original_intent,
                            deviation=deviation,
                            reason=reason,
                            result=result,
                            decision_factors=decision_factors or {}
                        )
                        session.add(deviation_entry)
                        await session.commit()
                        logger.info(f"[MEMORY] Synced deviation to SQL")
                
                # Check if we are in an event loop (likely)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_sync_to_sql())
                except RuntimeError:
                    # Not in loop, use run
                    asyncio.run(_sync_to_sql())
                    
            except Exception as sql_e:
                logger.warning(f"[MEMORY] Failed to sync deviation to SQL: {sql_e}")

            return True
        except Exception as e:
            logger.error(f"[MEMORY] Failed to store deviation: {e}")
            return False

    def recall_behavioral_logic(self, intent: str, n_results: int = 2) -> List[Dict[str, Any]]:
        """Recall past behavioral deviations for a given intent."""
        if not self.available or self.behavior_deviations.count() == 0: return []
        try:
            results = self.behavior_deviations.query(
                query_texts=[intent],
                n_results=min(n_results, self.behavior_deviations.count()),
                include=["documents", "metadatas", "distances"]
            )
            similar = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    similar.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 1.0
                    })
            return similar
        except Exception as e:
            logger.error(f"[MEMORY] Failed to recall deviations: {e}")
            return []


# Singleton instance
long_term_memory = LongTermMemory()
