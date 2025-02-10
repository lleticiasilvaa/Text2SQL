import sqlite3
import re
from typing import Dict, Any, List, Optional, TypedDict
from contextlib import contextmanager
from src.llm.llm import LLMManager, ModelConfig

SCHEMA_LINKING = "schema_linking"
TEXT_TO_SQL = "text_to_sql"

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()][:100]

class QueryGenerator:
    def __init__(
        self,
        llm_manager: LLMManager,
        schema: str,
        db_manager: DatabaseManager
    ):
        self.llm_manager = llm_manager
        self.schema = schema
        self.db_manager = db_manager
        self.schema_linking_system = "Given a user question and the schema of a database, your task is to generate an JSON with the names of tables and columns of the schema that the question is referring to."
        self.text_to_sql_system = "Given a user question and the schema of a database, your task is to generate an SQL query that accurately answers the question based on the provided schema."

    def schema_linking(self, question: str) -> Dict[str, Any]:
        try:
            self.llm_manager.set_adapter(SCHEMA_LINKING)
            messages = [
              {'role': 'system', 'content': self.schema_linking_system},
              {'role': 'user', 'content': f"# Schema:\n```sql\n{self.schema}\n```\n\n# Question: {question}"}
            ]
            response = self.llm_manager.generate_completion(messages, max_tokens=500)
            linked_schema_content = re.findall(r'```json(.*?)```', response, re.DOTALL)[0]
            return eval(linked_schema_content)
        except Exception as e:
            print(e)
            return {}

    def generate_sql(self, question: str, reduced_schema: str) -> str:
        try:
            self.llm_manager.set_adapter(TEXT_TO_SQL)
            messages = [
              {'role': 'system', 'content': self.text_to_sql_system},
              {'role': 'user', 'content': f"# Schema:\n```sql\n{reduced_schema}\n```\n\n# Question: {question}"}
            ]
            response = self.llm_manager.generate_completion(
                messages,
                max_tokens=500,
                temperature=0
            )
            return re.findall(r'```sql(.*?)```', response, re.DOTALL)[0]
        except Exception as e:
            print(e)
            return ""