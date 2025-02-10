from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import time
from dotenv import load_dotenv
import logging

from src.db.schema_linking import get_complete_schema, generate_reduced_schema_for_tables, SQLDatabase
from src.llm.llm import LLMManager, ModelConfig
from src.db.database import DatabaseManager, QueryGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(),
    logging.FileHandler("response.log", mode='a')
])
logger = logging.getLogger(__name__)

class Question(BaseModel):
    question: str


class SQLResponse(BaseModel):
    linked_schema: Dict[str, Any]
    sql_query: str
    sql_response: List[Dict[str, Any]]  
    schema_linking_time: float
    text_to_sql_time: float
    query_execution_time: float
    total_time: float


app = FastAPI()

# Load configuration
load_dotenv()
config = ModelConfig(
    model=os.getenv("MODEL", ""),
    lora_schema_linking=os.getenv("LORA_SCHEMA_LINKING", ""),
    lora_text_to_sql=os.getenv("LORA_TEXT_TO_SQL", ""),
    db_path=os.getenv("DB_PATH", ""),
    mode=os.getenv("MODE", "adapters"),
    quantize=True
)

# Initialize managers
db_manager = DatabaseManager(config['db_path'])
llm_manager = LLMManager(config)
schema = SQLDatabase(db_path=config['db_path'], num_examples=0)

query_generator = QueryGenerator(llm_manager, schema, db_manager)


@app.post("/text-to-sql/", response_model=SQLResponse)
async def convert_text_to_sql(question_input: Question) -> SQLResponse:
    try:
        #logger.info(f"Received question: {question_input.question}")
        
        # Step 1: Schema Linking
        start_time = time.time()
        schema_json = query_generator.schema_linking(question_input.question)
        linked_schema = generate_reduced_schema_for_tables(
            config['db_path'],
            schema_json,
            num_examples=0
        )
        schema_linking_time = time.time() - start_time

        # Step 2: Text-to-SQL
        start_time = time.time()
        sql_query = query_generator.generate_sql(question_input.question, linked_schema)
        text_to_sql_time = time.time() - start_time

        # Step 3: Execute Query
        start_time = time.time()
        try:
            result = db_manager.execute_query(sql_query)
        except Exception as e:
            result = [{"error": str(e)}]
            logger.error(f"Error executing query: {str(e)}")

        query_execution_time = time.time() - start_time
        
        total_time = schema_linking_time + text_to_sql_time + query_execution_time

        response_data = {
            "question": question_input.question,
            "linked_schema": schema_json,
            "sql_query": sql_query,
            "schema_linking_time": schema_linking_time,
            "text_to_sql_time": text_to_sql_time,
            "query_execution_time": query_execution_time,
            "total_time": total_time
        }
        logger.info(f"json: {response_data}")

        return SQLResponse(
            linked_schema=schema_json,
            sql_query=sql_query,
            sql_response=result,
            schema_linking_time=schema_linking_time,
            text_to_sql_time=text_to_sql_time,
            query_execution_time=query_execution_time,
            total_time=total_time
        )

        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app", 
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
