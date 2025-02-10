import sqlite3
from typing import Dict, List, Tuple

def get_primary_keys(columns_info: List[Tuple]) -> List[str]:
    """Extract primary keys from column info."""
    return [column[1].replace('"', '') for column in columns_info if column[5] == 1]

def get_foreign_keys(cursor, table_name: str) -> List[Tuple[str, str, str]]:
    """Fetch foreign key information for the given table."""
    cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
    return [(fk[3].replace('"', ''), fk[2].replace('"', ''), fk[4].replace('"', '')) for fk in cursor.fetchall()]

def generate_column_definitions(columns_info: List[Tuple[str, str, str]]) -> str:
    """Generate the column definitions for table creation."""
    column_defs = []
    for column in columns_info:
        column_name = column[1].replace('"', '')
        column_type = column[2].upper()
        column_defs.append(f'        {column_name} {column_type}')
    return ",\n".join(column_defs)

def generate_foreign_keys(foreign_keys: List[Tuple[str, str, str]]) -> str:
    """Generate foreign key definitions."""
    return "\n".join([f'        FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col})'
                      for fk_col, ref_table, ref_col in foreign_keys])

def add_sample_data(cursor, table_name: str, columns: List[Tuple]) -> str:
    """Add sample data to the schema."""
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
    rows = cursor.fetchall()
    sample_data = f"/*\n{len(rows)} rows from {table_name} table:\n"
    sample_data += "\t".join([col[1].lower().replace('"', '') for col in columns]) + "\n"
    for row in rows:
        sample_data += "\t".join(map(str, row)) + "\n"
    sample_data += "*/\n\n"
    return sample_data

def generate_table_creation(cursor, table_name: str, num_examples: int = 0) -> str:
    """Generate CREATE TABLE statement along with primary keys, foreign keys, and sample data."""
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    columns_info = cursor.fetchall()
    
    primary_keys = get_primary_keys(columns_info)
    foreign_keys = get_foreign_keys(cursor, table_name)
    column_defs = generate_column_definitions(columns_info)
    
    schema_str = f'CREATE TABLE {table_name} (\n{column_defs}'
    
    if primary_keys:
        schema_str += f',\n        PRIMARY KEY ({", ".join(primary_keys)})'

    if foreign_keys:
        schema_str += f',\n{generate_foreign_keys(foreign_keys)}'

    schema_str += "\n);\n\n"

    if num_examples > 0:
        schema_str += add_sample_data(cursor, table_name, columns_info)
    
    return schema_str

def generate_reduced_schema_for_tables(db_path: str, table_names: List[str], num_examples: int = 0) -> str:
    """Generate reduced schema for specified tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_str = ""
    for table_name in table_names:
        schema_str += generate_table_creation(cursor, table_name, num_examples)

    conn.close()
    return schema_str

def generate_reduced_schema(db_path: str, tables_and_columns: Dict[str, List[str]], num_examples: int = 0) -> str:
    """Generate reduced schema for specified tables and columns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_str = ""

    for table_name, columns in tables_and_columns.items():
        # Convert columns to lowercase and prepare for inclusion
        columns_to_include = [col.strip().lower().replace('"', '') for col in columns]

        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns_info = cursor.fetchall()

        # Filter columns to include based on the provided list
        included_columns = [column for column in columns_info if column[1].lower().replace('"', '') in columns_to_include]

        primary_keys = get_primary_keys(columns_info)
        foreign_keys = get_foreign_keys(cursor, table_name)
        column_defs = generate_column_definitions(included_columns)

        schema_str += f'CREATE TABLE {table_name} (\n{column_defs}'
        
        if primary_keys:
            schema_str += f',\n        PRIMARY KEY ({", ".join(primary_keys)})'

        if foreign_keys:
            schema_str += f',\n{generate_foreign_keys(foreign_keys)}'

        schema_str += "\n);\n\n"

        if num_examples > 0:
            schema_str += add_sample_data(cursor, table_name, included_columns)

    conn.close()
    return schema_str

def get_schema(json: dict, db_path: str, only_tables: bool = True) -> str:
    """Get schema based on whether only tables or full schema is needed."""
    if only_tables:
        tables = list(json.keys())
        return generate_reduced_schema_for_tables(db_path, tables, 0)
    else:
        return generate_reduced_schema(db_path, json, 0)

def get_complete_schema(db_path: str, num_examples: int = 0) -> str:
    """Generate complete schema for the entire database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_str = ""
    for table in tables:
        table_name = table[0]
        schema_str += generate_table_creation(cursor, table_name, num_examples)

    conn.close()
    return schema_str



def SQLDatabase(db_path, num_examples = 0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_str = ""

    # Obter uma lista de todas as tabelas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:

        table_name=table[0]
        # obter informações das colunas da tabela
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        included_columns = cursor.fetchall()

        schema_str += f'CREATE TABLE {table_name.lower()} (\n'

        primary_keys = []
        for column in included_columns:
            column_name = column[1].replace('"','')
            column_type = column[2]
            schema_str += f'        {column_name.lower()} {column_type.upper()},\n'

            if column[5] == 1:
                primary_keys.append(column[1].replace('"',''))

        schema_str = schema_str.rstrip(",\n") # remover a última vírgula e nova linha extra

        # Adicionar chaves primárias ao esquema
        if primary_keys:
            primary_keys_str = [pk.replace('"','').lower() for pk in primary_keys]
            primary_keys_str = ", ".join(primary_keys_str)
            schema_str += f',\n        PRIMARY KEY ({primary_keys_str})'

        # Adicionar definições de chave estrangeira ao esquema
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys_info = cursor.fetchall()
        for fk in foreign_keys_info:
          try:
              fk_col = fk[3].replace('"','')          # Coluna com chave estrangeira
              ref_table = fk[2].replace('"','')       # Tabela referenciada
              ref_col = fk[4].replace('"','')         # Coluna referenciada
              schema_str += f',\n        FOREIGN KEY ({fk_col.lower()}) REFERENCES {ref_table.lower()}({ref_col.lower()})'
          except:
            print(fk)

        schema_str += "\n);\n\n"

        if num_examples > 0:
          # Adicionar exemplos de dados
          cursor.execute(f"SELECT {', '.join([col[1] for col in included_columns])} FROM {table_name} LIMIT {num_examples};")
          rows = cursor.fetchall()
          # Adicionar dados de exemplo
          schema_str += f"/*\n{len(rows)} rows from {table_name} table:\n"
          schema_str += "\t".join([col[1].lower().replace('"','') for col in included_columns]) + "\n"
          for row in rows:
              schema_str += "\t".join(map(str, row)) + "\n"
          schema_str += "*/\n\n"

    schema_str = schema_str.rstrip('\n\n')

    # Fechar a conexão
    conn.close()
    return schema_str