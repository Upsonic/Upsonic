import psycopg
import json

from psycopg.errors import OperationalError, InFailedSqlTransaction
from psycopg.sql import SQL, Identifier, Literal as SqlLiteral, Composed, Placeholder

from typing import Any, Dict, List, Optional, Union, Literal as TypingLiteral, Tuple

from upsonic.vectordb.config import (
    Config,
    ProviderName,
    IndexType,
    DistanceMetric,
    HNSWTuningConfig,
    IVFTuningConfig,
    FlatTuningConfig
)
from upsonic.vectordb.base import BaseVectorDBProvider

from upsonic.utils.package.exception import(
    VectorDBConnectionError, 
    ConfigurationError, 
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError
)

from upsonic.schemas.vector_schemas import VectorSearchResult

class PgvectorProvider(BaseVectorDBProvider):
    """
    An implementation of the BaseVectorDBProvider for PostgreSQL with the pgvector extension.
    
    This provider translates the abstract framework configurations and calls into
    precise, parameterized SQL statements. It manages the connection lifecycle,
    table and index schema, data manipulation, and query generation required
    to use PostgreSQL as a powerful vector database.
    """

    def __init__(self, config: Config):
        if config.core.provider_name.value != 'pgvector':
            raise ConfigurationError(
                f"Attempted to initialize PgvectorProvider with a configuration for "
                f"'{config.core.provider_name.value}'."
            )
        super().__init__(config)
        self._connection: Optional[psycopg.Connection] = None
        print(f"PgvectorProvider initialized for table '{self._config.core.collection_name}'.")

    def connect(self) -> None:
        if self._is_connected and self._connection:
            print("Already connected to PostgreSQL.")
            return

        print(f"Attempting to connect to PostgreSQL at {self._config.core.host}:{self._config.core.port}...")
        try:
            conninfo = (
                f"host={self._config.core.host} "
                f"port={self._config.core.port} "
                f"dbname={self._config.core.db_path} "
                f"user={self._config.advanced.namespace or 'postgres'} "
                f"password={self._config.core.api_key.get_secret_value() if self._config.core.api_key else ''}"
            )
            
            self._connection = psycopg.connect(conninfo)
            if not self._connection:
                raise OperationalError("psycopg.connect() returned None.")
            
            with self._connection.cursor() as cursor:
                print("Verifying 'vector' extension exists in PostgreSQL...")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("'vector' extension is enabled.")
            
            self._connection.commit()
            self._is_connected = True
            print("Successfully connected to PostgreSQL.")
        except OperationalError as e:
            self._connection = None
            self._is_connected = False
            raise VectorDBConnectionError(f"Failed to connect to PostgreSQL: {e}")
        except Exception as e:
            self._connection = None
            self._is_connected = False
            raise VectorDBConnectionError(f"An unexpected error occurred during connection: {e}")

    def disconnect(self) -> None:
        if self._is_connected and self._connection:
            try:
                self._connection.close()
                self._is_connected = False
                self._connection = None
                print("Successfully disconnected from PostgreSQL.")
            except Exception as e:
                self._is_connected = False
                self._connection = None
                print(f"An error occurred during disconnection, but status is now 'disconnected'. Error: {e}")
        else:
            print("Already disconnected. No action taken.")

    def is_ready(self) -> bool:
        if not self._is_connected or not self._connection or self._connection.closed:
            return False
        
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                return result[0] == 1
        except (OperationalError, psycopg.InterfaceError):
            self._is_connected = False
            return False

    def create_collection(self) -> None:
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before creating a collection.")

        table_name = self._config.core.collection_name

        try:
            if self.collection_exists():
                if self._config.core.recreate_if_exists:
                    print(f"Table '{table_name}' already exists. Deleting and recreating as requested.")
                    self.delete_collection()
                else:
                    print(f"Table '{table_name}' already exists and 'recreate_if_exists' is False. No action taken.")
                    return

            with self._connection.cursor() as cursor:
                create_table_sql = SQL("""
                    CREATE TABLE {table} (
                        id TEXT PRIMARY KEY,
                        payload JSONB,
                        vector vector({vector_size}),
                        tenant_id TEXT NOT NULL
                    );
                """).format(
                    table=Identifier(table_name),
                    vector_size=SqlLiteral(self._config.core.vector_size)
                )
                print(f"Creating table '{table_name}'...")
                cursor.execute(create_table_sql)

                print("Creating indexes...")
                tenant_index_sql = SQL("CREATE INDEX ON {table} (tenant_id);").format(table=Identifier(table_name))
                cursor.execute(tenant_index_sql)
                
                payload_index_sql = SQL("CREATE INDEX ON {table} USING gin (payload);").format(table=Identifier(table_name))
                cursor.execute(payload_index_sql)

                index_conf = self._config.indexing.index_config
                if isinstance(index_conf, (HNSWTuningConfig, IVFTuningConfig)):
                    distance_map = {
                        DistanceMetric.COSINE: "vector_cosine_ops",
                        DistanceMetric.EUCLIDEAN: "vector_l2_ops",
                        DistanceMetric.DOT_PRODUCT: "vector_ip_ops",
                    }
                    operator_class = distance_map.get(self._config.core.distance_metric)
                    if not operator_class:
                        raise ConfigurationError(f"Unsupported distance metric '{self._config.core.distance_metric}' for pgvector.")
                    
                    if isinstance(index_conf, HNSWTuningConfig):
                        index_type, with_opts = "hnsw", f"m = {index_conf.m}, ef_construction = {index_conf.ef_construction}"
                    else:
                        index_type, with_opts = "ivfflat", f"lists = {index_conf.nlist}"

                    vector_index_sql = SQL("""
                        CREATE INDEX ON {table} USING {index_type} (vector {operator_class}) 
                        WITH ({with_opts});
                    """).format(
                        table=Identifier(table_name), index_type=SQL(index_type),
                        operator_class=SQL(operator_class), with_opts=SQL(with_opts)
                    )
                    print(f"Creating {index_type.upper()} index on 'vector' column...")
                    cursor.execute(vector_index_sql)

                elif isinstance(index_conf, FlatTuningConfig):
                    print("Index type is FLAT. No vector index created.")
            
            self._connection.commit()
            print(f"Successfully created table '{table_name}' and its indexes.")
        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise VectorDBError(f"Failed to create collection '{table_name}': {e}")

    def delete_collection(self) -> None:
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before deleting a collection.")
        
        table_name = self._config.core.collection_name
        
        try:
            with self._connection.cursor() as cursor:
                delete_sql = SQL("DROP TABLE IF EXISTS {table} CASCADE;").format(table=Identifier(table_name))
                print(f"Deleting table '{table_name}'...")
                cursor.execute(delete_sql)
            
            self._connection.commit()
            print(f"Successfully deleted table '{table_name}'.")
        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise VectorDBError(f"Failed to delete collection '{table_name}': {e}")

    def collection_exists(self) -> bool:
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL to check for a collection's existence.")
        
        try:
            with self._connection.cursor() as cursor:
                exists_sql = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);"
                cursor.execute(exists_sql, (self._config.core.collection_name,))
                result = cursor.fetchone()
                return result[0] if result else False
        except (Exception, psycopg.Error) as e:
            if isinstance(e, InFailedSqlTransaction) and self._connection:
                self._connection.rollback()
            print(f"Could not check for collection existence: {e}")
            return False

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[Union[str, int]], chunks: Optional[List[str]], **kwargs) -> None:
        """
        Adds new data or updates existing data in the table using PostgreSQL's
        `INSERT ... ON CONFLICT` statement for atomic and efficient upserts.

        Args:
            vectors: A list of vector embeddings.
            payloads: A list of corresponding metadata objects (will be stored in JSONB).
            ids: A list of unique identifiers for each vector-payload pair.

        Raises:
            UpsertError: If the data ingestion fails.
            ConfigurationError: If no `namespace` is provided in the config, as it's
                                required for the `tenant_id` column.
        """
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before upserting data.")
        
        if not (len(vectors) == len(payloads) == len(ids)):
            raise UpsertError("The lengths of vectors, payloads, and ids lists must be identical.")
        if not vectors:
            print("Upsert called with empty lists. No action taken.")
            return

        if chunks is not None:
            if len(chunks) != len(payloads):
                raise UpsertError("The lengths of chunks and payloads lists must be identical.")
            for i in range(len(payloads)):
                payloads[i]["chunk"] = chunks[i]

        tenant_id = self._config.advanced.namespace
        if not tenant_id:
            raise ConfigurationError("A `namespace` must be provided in the AdvancedConfig for pgvector multi-tenancy.")

        table_name = self._config.core.collection_name
        
        try:
            data_to_insert = [
                (str(ids[i]), json.dumps(payloads[i]), vectors[i], tenant_id)
                for i in range(len(vectors))
            ]

            upsert_sql = SQL("""
                INSERT INTO {table} (id, payload, vector, tenant_id) VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    vector = EXCLUDED.vector;
            """).format(table=Identifier(table_name))

            with self._connection.cursor() as cursor:
                print(f"Upserting {len(data_to_insert)} records into '{table_name}'...")
                cursor.executemany(upsert_sql, data_to_insert)
            
            self._connection.commit()
            print("Upsert successful.")

        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise UpsertError(f"Failed to upsert data to '{table_name}': {e}")


    def delete(self, ids: List[Union[str, int]], **kwargs) -> None:
        """
        Removes data from the collection by their unique identifiers, scoped to the
        provider's configured tenant.

        Args:
            ids: A list of specific IDs to remove.
        """
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before deleting data.")
        if not ids:
            print("Delete called with an empty list of IDs. No action taken.")
            return

        tenant_id = self._config.advanced.namespace
        if not tenant_id:
            raise ConfigurationError("A `namespace` must be provided in the AdvancedConfig for pgvector multi-tenancy.")

        table_name = self._config.core.collection_name

        try:
            delete_sql = SQL("DELETE FROM {table} WHERE tenant_id = %s AND id = ANY(%s);").format(
                table=Identifier(table_name)
            )
            
            with self._connection.cursor() as cursor:
                cursor.execute(delete_sql, (tenant_id, [str(id_val) for id_val in ids]))
                print(f"Delete command affected {cursor.rowcount} rows in '{table_name}'.")

            self._connection.commit()
        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise VectorDBError(f"Failed to delete data from '{table_name}': {e}")

    
    def fetch(self, ids: List[Union[str, int]], **kwargs) -> List[VectorSearchResult]:
        """
        Retrieves full records (payload and vector) by their IDs, scoped to the
        provider's configured tenant.

        Args:
            ids: A list of IDs to retrieve the full records for.

        Returns:
            A list of VectorSearchResult objects containing the fetched data.
        """
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before fetching data.")
        if not ids:
            return []

        tenant_id = self._config.advanced.namespace
        if not tenant_id:
            raise ConfigurationError("A `namespace` must be provided in the AdvancedConfig for pgvector multi-tenancy.")
        
        table_name = self._config.core.collection_name
        results = []

        try:
            fetch_sql = SQL("SELECT id, payload, vector FROM {table} WHERE tenant_id = %s AND id = ANY(%s);").format(
                table=Identifier(table_name)
            )

            with self._connection.cursor() as cursor:
                cursor.execute(fetch_sql, (tenant_id, [str(id_val) for id_val in ids]))
                
                for row in cursor.fetchall():
                    payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                    
                    if hasattr(row[2], 'tolist'):
                        vector = row[2].tolist()
                    elif isinstance(row[2], (list, tuple)):
                        vector = list(row[2])
                    elif isinstance(row[2], str):
                        import ast
                        try:
                            vector = ast.literal_eval(row[2])
                        except:
                            vector = row[2]
                    else:
                        vector = row[2]
                    
                    text = payload.get("chunk", "") if payload else ""
                    
                    results.append(VectorSearchResult(
                        id=row[0],
                        score=1.0,
                        payload=payload,
                        vector=vector,
                        text=text
                    ))
            return results
        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise VectorDBError(f"Failed to fetch data from '{table_name}': {e}")


    def search(self, top_k: Optional[int] = None, query_vector: Optional[List[float]] = None, query_text: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[TypingLiteral['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        
        filter = filter if filter is not None else self._config.search.filter
        final_top_k = top_k if top_k is not None else self._config.search.default_top_k or 10
        has_vector = query_vector is not None and len(query_vector) > 0
        has_text = query_text is not None and query_text.strip()

        if has_vector and not has_text:
            if self._config.search.dense_search_enabled is False:
                raise ConfigurationError("Dense search is disabled by the current configuration.")
            print("Dispatching to: dense_search")
            return self.dense_search(query_vector=query_vector, top_k=final_top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)
        
        elif not has_vector and has_text:
            if self._config.search.full_text_search_enabled is False:
                raise ConfigurationError("Full-text search is disabled by the current configuration.")
            print("Dispatching to: full_text_search")
            return self.full_text_search(query_text=query_text, top_k=final_top_k, filter=filter, similarity_threshold=similarity_threshold, **kwargs)

        elif has_vector and has_text:
            if self._config.search.hybrid_search_enabled is False:
                raise ConfigurationError("Hybrid search is disabled by the current configuration.")
            print("Dispatching to: hybrid_search")
            return self.hybrid_search(query_vector=query_vector, query_text=query_text, top_k=final_top_k, filter=filter, alpha=alpha, fusion_method=fusion_method, similarity_threshold=similarity_threshold, **kwargs)
        else:
            raise ConfigurationError("Search requires at least one of 'query_vector' or 'query_text'.")


    def dense_search(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before searching.")
        
        tenant_id = self._config.advanced.namespace
        if not tenant_id:
            raise ConfigurationError("A `namespace` must be provided for pgvector multi-tenancy.")

        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5

        table_name = self._config.core.collection_name
        
        metric_map = {
            DistanceMetric.COSINE: (SQL("<=>"), SQL("1 - (vector <=> %s::vector) AS score"), "ASC"),
            DistanceMetric.EUCLIDEAN: (SQL("<->"), SQL("vector <-> %s::vector AS score"), "ASC"),
            DistanceMetric.DOT_PRODUCT: (SQL("<#>"), SQL("(vector <#> %s::vector) * -1 AS score"), "DESC"),
        }
        
        metric = self._config.core.distance_metric
        if metric not in metric_map:
            raise ConfigurationError(f"Unsupported distance metric for dense search: {metric}")

        distance_op, score_sql, order_direction = metric_map[metric]

        query_parts = [
            SQL("SELECT id, payload, vector,"),
            score_sql,
            SQL("FROM {} WHERE tenant_id = %s").format(Identifier(table_name))
        ]

        params = [query_vector, tenant_id]

        if filter:
            filter_sql, filter_params = self._translate_filter(filter)
            query_parts.append(SQL("AND"))
            query_parts.append(filter_sql)
            params.extend(filter_params)

        query_parts.append(SQL("ORDER BY vector ") + distance_op + SQL(" %s::vector ") + SQL(order_direction))
        params.append(query_vector)

        query_parts.append(SQL("LIMIT %s"))
        params.append(top_k)

        final_sql = SQL(" ").join(query_parts)
        
        results = []
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(final_sql, params)
                for row in cursor.fetchall():
                    payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                    
                    if hasattr(row[2], 'tolist'):
                        vector = row[2].tolist()
                    elif isinstance(row[2], (list, tuple)):
                        vector = list(row[2])
                    elif isinstance(row[2], str):
                        import ast
                        try:
                            vector = ast.literal_eval(row[2])
                        except:
                            vector = row[2]
                    else:
                        vector = row[2]
                    
                    text = payload.get("chunk", "") if payload else ""
                    score = float(row[3])
                    
                    if score >= final_similarity_threshold:
                        results.append(VectorSearchResult(
                            id=row[0],
                            payload=payload,
                            vector=vector,
                            score=score,
                            text=text
                        ))
            return results
        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise SearchError(f"An error occurred during dense search: {e}")

    def full_text_search(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        if not self._is_connected or not self._connection:
            raise VectorDBConnectionError("Must be connected to PostgreSQL before searching.")

        tenant_id = self._config.advanced.namespace
        if not tenant_id:
            raise ConfigurationError("A `namespace` must be provided for pgvector multi-tenancy.")
        
        final_similarity_threshold = similarity_threshold if similarity_threshold is not None else self._config.search.default_similarity_threshold or 0.5
        
        fts_field = kwargs.get("fts_field")
        if not fts_field:
            raise ConfigurationError("`fts_field` must be provided in kwargs for full_text_search.")

        table_name = self._config.core.collection_name
        
        query_parts = [
            SQL("SELECT id, payload, vector,"),
            SQL("ts_rank(to_tsvector('english', payload->>%s), plainto_tsquery('english', %s)) AS score"),
            SQL("FROM {table} WHERE to_tsvector('english', payload->>%s) @@ plainto_tsquery('english', %s) AND tenant_id = %s").format(
                table=Identifier(table_name)
            )
        ]
        
        params = [fts_field, query_text, fts_field, query_text, tenant_id]

        if filter:
            filter_sql, filter_params = self._translate_filter(filter)
            query_parts.append(SQL("AND"))
            query_parts.append(filter_sql)
            params.extend(filter_params)

        query_parts.append(SQL("ORDER BY score DESC LIMIT %s"))
        params.append(top_k)

        final_sql = SQL(" ").join(query_parts)
        results = []
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(final_sql, params)
                for row in cursor.fetchall():
                    payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                    
                    if hasattr(row[2], 'tolist'):
                        vector = row[2].tolist()
                    elif isinstance(row[2], (list, tuple)):
                        vector = list(row[2])
                    elif isinstance(row[2], str):
                        import ast
                        try:
                            vector = ast.literal_eval(row[2])
                        except:
                            vector = row[2]
                    else:
                        vector = row[2]
                    
                    text = payload.get("chunk", "") if payload else ""
                    score = float(row[3])
                    
                    if score >= final_similarity_threshold:
                        results.append(VectorSearchResult(
                            id=row[0], 
                            payload=payload, 
                            vector=vector, 
                            score=score,
                            text=text
                        ))
            return results
        except (Exception, psycopg.Error) as e:
            if self._connection: self._connection.rollback()
            raise SearchError(f"An error occurred during full-text search: {e}")

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None, alpha: Optional[float] = None, fusion_method: Optional[TypingLiteral['rrf', 'weighted']] = None, similarity_threshold: Optional[float] = None, **kwargs) -> List[VectorSearchResult]:
        """
        Performs a hybrid search by fetching results from dense and full-text searches
        and fusing them using a Reciprocal Rank Fusion (RRF) algorithm.
        """
        final_alpha = alpha if alpha is not None else self._config.search.default_hybrid_alpha or 0.5
        fusion_k = 60

        candidate_pool_size = top_k

        dense_results = self.dense_search(query_vector, candidate_pool_size, filter, similarity_threshold, **kwargs)

        keyword_results = self.full_text_search(query_text, candidate_pool_size, filter, similarity_threshold, **kwargs)

        all_docs = {doc.id: doc for doc in dense_results}
        all_docs.update({doc.id: doc for doc in keyword_results})

        rrf_scores = {}
        
        for i, doc in enumerate(dense_results):
            if doc.id not in rrf_scores:
                rrf_scores[doc.id] = 0
            rrf_scores[doc.id] += final_alpha * (1 / (fusion_k + i + 1))
            
        for i, doc in enumerate(keyword_results):
            if doc.id not in rrf_scores:
                rrf_scores[doc.id] = 0
            rrf_scores[doc.id] += (1 - final_alpha) * (1 / (fusion_k + i + 1))
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)

        final_results = []
        for doc_id in sorted_ids[:top_k]:
            result = all_docs[doc_id]
            result_with_fused_score = VectorSearchResult(
                id=result.id,
                payload=result.payload,
                vector=result.vector,
                score=rrf_scores[doc_id],
                text=result.text
            )
            final_results.append(result_with_fused_score)
            
        return final_results
    


    def _translate_filter(self, filter_dict: Dict[str, Any]) -> Tuple[Composed, List[Any]]:
        """
        Recursively translates a framework-standard filter dictionary into a
        parameterized psycopg SQL Composed object and a list of parameters.
        """
        sql_parts = []
        params = []

        logical_ops = {"and": SQL(" AND "), "or": SQL(" OR ")}

        if len(filter_dict) == 1 and list(filter_dict.keys())[0] in logical_ops:
            op = list(filter_dict.keys())[0]
            sub_filters = filter_dict[op]

            sub_sql_parts = []
            for sub_filter in sub_filters:
                sub_sql, sub_params = self._translate_filter(sub_filter)
                sub_sql_parts.append(sub_sql)
                params.extend(sub_params)

            sql_composition = SQL("(") + logical_ops[op].join(sub_sql_parts) + SQL(")")
            return sql_composition, params

        for field, value in filter_dict.items():
            field_sql = SQL("(payload->>") + Placeholder() + SQL(")")
            params.append(field)

            if isinstance(value, dict):
                op, val = list(value.items())[0]
                if op == "$gte":
                    sql_parts.append(field_sql + SQL("::numeric >= ") + Placeholder())
                elif op == "$lte":
                    sql_parts.append(field_sql + SQL("::numeric <= ") + Placeholder())
                elif op == "$gt":
                    sql_parts.append(field_sql + SQL("::numeric > ") + Placeholder())
                elif op == "$lt":
                    sql_parts.append(field_sql + SQL("::numeric < ") + Placeholder())
                elif op == "$ne":
                    sql_parts.append(field_sql + SQL("::text != ") + Placeholder())
                elif op == "$in":
                    sql_parts.append(field_sql + SQL("::text = ANY(") + Placeholder() + SQL(")"))
                else:
                    raise SearchError(f"Unsupported filter operator '{op}'.")
                params.append(val)
            else:
                sql_parts.append(field_sql + SQL("::text = ") + Placeholder())
                params.append(str(value))

        return SQL(" AND ").join(sql_parts), params