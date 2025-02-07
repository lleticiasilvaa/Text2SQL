import sqlite3

def SQLDatabase(db_path, num_examples = 0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_str = ""

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:

        table_name=table[0]
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

        schema_str = schema_str.rstrip(",\n") 

        # Adicionar chaves primárias ao esquema
        if primary_keys:
            primary_keys_str = [pk.replace('"','').lower() for pk in primary_keys]
            primary_keys_str = ", ".join(primary_keys_str)
            schema_str += f',\n        PRIMARY KEY ({primary_keys_str})'


        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys_info = cursor.fetchall()
        for fk in foreign_keys_info:
          try:
              fk_col = fk[3].replace('"','')          
              ref_table = fk[2].replace('"','')       
              ref_col = fk[4].replace('"','')         
              schema_str += f',\n        FOREIGN KEY ({fk_col.lower()}) REFERENCES {ref_table.lower()}({ref_col.lower()})'
          except:
            print(fk)

        schema_str += "\n);\n\n"

        if num_examples > 0:
          cursor.execute(f"SELECT {', '.join([col[1] for col in included_columns])} FROM {table_name} LIMIT {num_examples};")
          rows = cursor.fetchall()
          schema_str += f"/*\n{len(rows)} rows from {table_name} table:\n"
          schema_str += "\t".join([col[1].lower().replace('"','') for col in included_columns]) + "\n"
          for row in rows:
              schema_str += "\t".join(map(str, row)) + "\n"
          schema_str += "*/\n\n"

    schema_str = schema_str.rstrip('\n\n')

    conn.close()
    return schema_str

def SQLDatabase_min(db_path, num_examples = 0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_str = ""

    # Obter uma lista de todas as tabelas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    #print(tables)

    for table in tables:

        table_name=table[0]

        # obter informações das colunas da tabela
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        included_columns = cursor.fetchall()

        schema_str += f'CREATE TABLE {table_name.lower()} (\n'

        primary_keys = []
        for column in included_columns:
            column_name = column[1].lower().replace('"','')
            column_type = column[2]
            schema_str += f'        {column_name} {column_type.upper()},\n'

            if column[5] == 1:
                primary_keys.append(column[1].lower().replace('"',''))

        schema_str = schema_str.rstrip(",\n") # remover a última vírgula e nova linha extra

        # Adicionar chaves primárias ao esquema
        if primary_keys:
            primary_keys_str = [pk.lower().replace('"','') for pk in primary_keys]
            primary_keys_str = ", ".join(primary_keys_str)
            schema_str += f',\n        PRIMARY KEY ({primary_keys_str})'

        # Adicionar definições de chave estrangeira ao esquema
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys_info = cursor.fetchall()
        for fk in foreign_keys_info:
            fk_col = fk[3].lower().replace('"','')          # Coluna com chave estrangeira
            ref_table = fk[2].lower().replace('"','')       # Tabela referenciada
            ref_col = fk[4].lower().replace('"','')         # Coluna referenciada
            schema_str += f',\n        FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col})'

        schema_str += "\n);\n\n"

        if num_examples > 0:
          # Adicionar exemplos de dados
          cursor.execute(f"SELECT {', '.join([col[1] for col in included_columns])} FROM {table_name} LIMIT {num_examples};")
          rows = cursor.fetchall()
          # Adicionar dados de exemplo
          schema_str += f"/*\n{len(rows)} rows from {table_name.lower()} table:\n"
          schema_str += "\t".join([col[1].lower().replace('"','') for col in included_columns]) + "\n"
          for row in rows:
              schema_str += "\t".join(map(str, row)) + "\n"
          schema_str += "*/\n\n"

    schema_str = schema_str.rstrip('\n\n')

    # Fechar a conexão
    conn.close()
    return schema_str

def schema_reduzido(db_path, tables_and_columns, num_examples = 0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables_and_columns_dict = eval(tables_and_columns)
    tables_and_columns_dict_min = eval(tables_and_columns.lower())

    schema_str = ""

    for table_name, columns in tables_and_columns_dict.items():

        table_name = table_name.replace('"','')
        columns_to_include = [col.strip().lower().replace('"','') for col in columns]

        # obter informações das colunas da tabela
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns_info = cursor.fetchall()

        included_columns = []
        primary_keys = []
        for column in columns_info:
            column_name = column[1].lower().replace('"','') #usar lower pra não ter interferencia de letra maiscula
            if column_name in columns_to_include:
                included_columns.append(column)

            if column[5] == 1:  # se a coluna for uma chave primária, sempre incluir
                primary_keys.append(column[1].replace('"',''))
                if column_name not in columns_to_include:
                    included_columns.append(column)

        schema_str += f'CREATE TABLE {table_name} (\n'
        for column in included_columns:
            column_name = column[1].replace('"','')
            column_type = column[2]
            schema_str += f'        {column_name} {column_type.upper()},\n'

        schema_str = schema_str.rstrip(",\n") # remover a última vírgula e nova linha extra

        # Adicionar chave primária ao esquema
        if primary_keys:
            primary_keys_str = [pk.replace('"','') for pk in primary_keys]
            primary_keys_str = ", ".join(primary_keys_str)
            schema_str += f',\n        PRIMARY KEY ({primary_keys_str})'

        # Adicionar definições de chave estrangeira ao esquema
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys_info = cursor.fetchall()

        for fk in foreign_keys_info:
            foreign_key = fk[3].replace('"','')  # Nome da coluna da chave estrangeira
            if foreign_key.lower() in columns_to_include: #confere se a coluna vai ser adicionada

                fk_col = fk[3].replace('"','')          # Coluna com chave estrangeira
                ref_table = fk[2].replace('"','')       # Tabela referenciada
                ref_col = fk[4].replace('"','')         # Coluna referenciada
                #if ref_table.lower() in tables_and_columns_dict_min.keys():
                #  if ref_col.lower() in tables_and_columns_dict_min[ref_table.lower()]:
                schema_str += f',\n        FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col})'

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

def schema_reduzido_tabelas(db_path, table_names, num_examples = 0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables_name_min = [table.lower() for table in table_names]

    schema_str = ""

    for table_name in table_names:

        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns_info = cursor.fetchall()

        included_columns = []
        primary_keys = []
        for column in columns_info:
            included_columns.append(column)
            if column[5] == 1:  # se a coluna for uma chave primária, sempre incluir
                primary_keys.append(column)


        schema_str += f'CREATE TABLE {table_name} (\n'
        for column in included_columns:
            column_name = column[1].replace('"','')
            column_type = column[2].upper()
            schema_str += f'        {column_name} {column_type.upper()},\n'

        schema_str = schema_str.rstrip(",\n") # remover a última vírgula e nova linha extra

        # Adicionar chave primária ao esquema
        if primary_keys:
            primary_keys_str = [pk[1].replace('"','') for pk in primary_keys]
            primary_keys_str = ", ".join(primary_keys_str)
            schema_str += f',\n        PRIMARY KEY ({primary_keys_str})'

        # Adicionar definições de chave estrangeira ao esquema
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys_info = cursor.fetchall()
        for fk in foreign_keys_info:
            fk_col = fk[3].replace('"','')          # Coluna com chave estrangeira
            ref_table = fk[2].replace('"','')       # Tabela referenciada
            ref_col = fk[4].replace('"','')         # Coluna referenciada


            #if ref_table.lower() in tables_name_min:
            schema_str += f',\n        FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col})'

        schema_str += "\n);\n\n"

        if num_examples > 0:
          # Adicionar exemplos de dados
          cursor.execute(f"SELECT * FROM {table_name} LIMIT {num_examples};")
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