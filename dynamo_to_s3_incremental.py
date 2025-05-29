import sys
import datetime
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from awsgluedq.transforms import EvaluateDataQuality
from pyspark.sql.functions import lit, current_date, current_timestamp, col, hash, struct
from pyspark.sql.types import StringType, StructType, StructField, MapType
import uuid
import json
import os

# Función para obtener el ID de la cuenta AWS
def get_aws_account_id():
    sts_client = boto3.client('sts')
    return sts_client.get_caller_identity()['Account']

# Función para crear cliente S3 en la región específica
def get_s3_client(region):
    return boto3.client('s3', region_name=region)

# Parámetros requeridos para el job
required_params = [
    'JOB_NAME',
    'dynamo_table_name',          # Nombre de la tabla en DynamoDB
    'bucket_name',                # Nombre del bucket S3
    'folder_name',                # Nombre de la carpeta en S3
    'primary_key',                # Clave primaria de la tabla
    'catalog_database',           # Base de datos del catálogo
    'catalog_table_name',         # Nombre de la tabla en el catálogo
]

# Obtenemos los parámetros requeridos
args = getResolvedOptions(sys.argv, required_params)

# Inicializar el contexto y job de Glue - SOLO UNA VEZ
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Obtener valores de los parámetros requeridos
dynamo_table_name = args['dynamo_table_name']
bucket_name = args['bucket_name']
folder_name = args['folder_name']
primary_key = args['primary_key']
catalog_database = args['catalog_database']
catalog_table_name = args['catalog_table_name']

# Obtener región actual de ejecución de Glue
current_region = boto3.session.Session().region_name
dynamo_account_id = get_aws_account_id()

# Definir parámetros opcionales con valores predeterminados
dynamo_region = "us-east-2"  # Virginia para DynamoDB
s3_region = "us-east-2"      # Ohio para S3
s3_temp_bucket = f"aws-glue-assets-{dynamo_account_id}-{current_region}"
s3_temp_prefix = "temporary/ddbexport/"

# Sobreescribir con valores de argumentos opcionales si se proporcionan
if 'dynamo_region' in args:
    dynamo_region = args['dynamo_region']
if 's3_region' in args:
    s3_region = args['s3_region']
if 's3_temp_bucket' in args:
    s3_temp_bucket = args['s3_temp_bucket']
if 's3_temp_prefix' in args:
    s3_temp_prefix = args['s3_temp_prefix']

# Fecha y hora actual para nombrar la carpeta
now = datetime.datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
timestamp = now.strftime("%H-%M-%S")
current_date_str = now.strftime("%Y-%m-%d")

# Default ruleset para calidad de datos
DEFAULT_DATA_QUALITY_RULESET = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Construir las rutas
dynamo_db_table_arn = f"arn:aws:dynamodb:{dynamo_region}:{dynamo_account_id}:table/{dynamo_table_name}"
s3_target_path = f"s3://{bucket_name}/{folder_name}/"
s3_new_records_path = f"s3://{bucket_name}/{folder_name}/{year}/{month}/{day}/{timestamp}/"

print(f"Procesando tabla DynamoDB: {dynamo_table_name}")
print(f"Región de DynamoDB: {dynamo_region}, Cuenta: {dynamo_account_id}")
print(f"Región de S3 destino: {s3_region}")
print(f"Región donde se ejecuta Glue: {current_region}")
print(f"Usando clave primaria: {primary_key}")
print(f"Ruta base en S3: {s3_target_path}")
print(f"Ruta para nuevos registros: {s3_new_records_path}")

def get_unified_schema(df):
    """
    Analiza todo el DataFrame para determinar un esquema unificado que contenga todos los campos encontrados.
    Todos los campos serán convertidos a StringType para mantener consistencia.
    
    Args:
        df: DataFrame de Spark con los datos a analizar
        
    Returns:
        StructType: Esquema unificado que contiene todos los campos encontrados
    """
    # Recolectar todos los nombres de columnas de todas las filas
    all_columns = set()
    
    # Usamos el esquema existente como base y exploramos datos para campos adicionales
    existing_schema = df.schema
    for field in existing_schema:
        all_columns.add(field.name)
    
    # Si el DataFrame contiene campos de tipo Map o Struct, necesitamos explorar más
    sample_data = df.limit(1000).collect()  # Muestra para detectar campos adicionales
    
    for row in sample_data:
        for field in row.__fields__:
            if field not in all_columns:
                all_columns.add(field)
    
    print(f"Esquema unificado contiene {len(all_columns)} campos")
    
    # Crear un esquema donde todos los campos son StringType
    unified_schema = StructType([
        StructField(col_name, StringType(), True) for col_name in sorted(all_columns)
    ])
    
    return unified_schema

def enforce_unified_schema(df, unified_schema):
    """
    Aplica el esquema unificado al DataFrame, asegurando que todos los campos estén presentes
    y convertidos al tipo correcto (StringType).
    
    Args:
        df: DataFrame original
        unified_schema: Esquema unificado a aplicar
        
    Returns:
        DataFrame: Nuevo DataFrame con el esquema unificado aplicado
    """
    # Convertir todas las columnas existentes a StringType
    for field in df.schema:
        df = df.withColumn(field.name, col(field.name).cast("string"))
    
    # Añadir columnas faltantes con valor null
    for field in unified_schema:
        if field.name not in df.columns:
            df = df.withColumn(field.name, lit(None).cast("string"))
    
    # Seleccionar solo las columnas del esquema unificado en el orden correcto
    df = df.select([field.name for field in unified_schema])
    
    return df

def write_consistent_parquet_files(df, primary_key, target_path, max_records_per_file=1000):
    """
    Escribe archivos Parquet consistentes con un esquema unificado, agrupando registros
    para optimizar el número de archivos creados.
    
    Args:
        df: DataFrame con los datos a escribir
        primary_key: Clave primaria para nombrar los archivos
        target_path: Ruta S3 de destino
        max_records_per_file: Máximo de registros por archivo Parquet
    """
    # Obtener esquema unificado
    unified_schema = get_unified_schema(df)
    
    # Aplicar esquema unificado
    df = enforce_unified_schema(df, unified_schema)
    
    # Dividir en grupos para evitar demasiados archivos pequeños
    df = df.withColumn("file_group", 
                      (hash(col(primary_key)) % (df.count() / max_records_per_file + 1)))
    
    # Escribir los archivos Parquet
    print(f"Escribiendo archivos Parquet consistentes en {target_path}")
    df.write.partitionBy("file_group").mode("overwrite").parquet(target_path)
    
    print("Escritura completada con esquema unificado")

def write_by_pk_to_s3_parquet_safe(dynamic_frame, primary_key, target_path):
    """
    Versión mejorada que escribe archivos Parquet con esquema consistente.
    """
    df = dynamic_frame.toDF()
    
    # Usar la nueva implementación con esquema unificado
    write_consistent_parquet_files(df, primary_key, target_path)
    
    return df.count()  
    
    
def modulo_leer_datos_dynamo():
    """
    Lee datos de una tabla DynamoDB y los convierte a un formato adecuado para Spark.
    Maneja todos los tipos de datos nativos de DynamoDB y asegura que todos los campos
    se procesen correctamente.
    """
    print(f"Conectando a DynamoDB en región {dynamo_region}...")
    
    try:
        # Usar cliente de bajo nivel para obtener datos en formato nativo de DynamoDB
        dynamodb = boto3.client('dynamodb', region_name=dynamo_region)
        
        print(f"Escaneando tabla {dynamo_table_name}...")
        response = dynamodb.scan(TableName=dynamo_table_name)
        items = response['Items']
        
        # Obtener todos los items en caso de paginación
        while 'LastEvaluatedKey' in response:
            print(f"Recuperando más registros (paginación)...")
            response = dynamodb.scan(
                TableName=dynamo_table_name,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response['Items'])
        
        # Mostrar información de diagnóstico
        print(f"Total de registros recuperados: {len(items)}")
        if items:
            print(f"Ejemplo del primer item (primeros 3 campos):")
            sample_item = items[0]
            sample_keys = list(sample_item.keys())[:3]
            for key in sample_keys:
                print(f"  {key}: {sample_item[key]}")
        
        # Procesar los items para convertir todo a strings
        processed_items = []
        for item in items:
            processed_item = {}
            for key, value in item.items():
                if 'S' in value:  # String
                    processed_item[key] = value['S']
                elif 'N' in value:  # Number
                    processed_item[key] = value['N']
                elif 'BOOL' in value:  # Boolean
                    # Convertir a 'true' o 'false' (minúsculas) para consistencia
                    processed_item[key] = str(value['BOOL']).lower()
                elif 'L' in value:  # List
                    processed_item[key] = str(value['L'])
                elif 'M' in value:  # Map
                    processed_item[key] = str(value['M'])
                elif 'NULL' in value:  # Null
                    processed_item[key] = "null"
                elif 'B' in value:  # Binary
                    processed_item[key] = "[binary data]"
                elif 'SS' in value:  # String Set
                    processed_item[key] = str(value['SS'])
                elif 'NS' in value:  # Number Set
                    processed_item[key] = str(value['NS'])
                elif 'BS' in value:  # Binary Set
                    processed_item[key] = "[binary set data]"
                else:
                    # Para cualquier otro formato no reconocido
                    processed_item[key] = str(value)
            
            processed_items.append(processed_item)
        
        # Determinar un schema común basado en todos los campos encontrados
        all_fields = set()
        for item in processed_items:
            all_fields.update(item.keys())
        
        print(f"Campos encontrados en la tabla: {', '.join(sorted(all_fields))}")
        
        # Asegurarse de que cada item tenga todos los campos (con valores nulos si es necesario)
        for item in processed_items:
            for field in all_fields:
                if field not in item:
                    item[field] = None
        
        # Crear un DataFrame con todos los campos como StringType
        if processed_items:
            schema_fields = [StructField(key, StringType(), True) for key in all_fields]
            schema = StructType(schema_fields)
            
            print(f"Creando DataFrame con {len(schema_fields)} columnas...")
            items_df = spark.createDataFrame(processed_items, schema=schema)
            
            # Convertir a DynamicFrame y retornar
            return DynamicFrame.fromDF(items_df, glueContext, "dynamo_dyf")
        else: 
            raise Exception("No hay elementos en la tabla para procesar")
    
    except Exception as e:
        print(f"Error al leer datos de DynamoDB: {str(e)}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Error al procesar DynamoDB: {str(e)}")

def evaluarCalidadDatos(dynamo_dyf):
    # Evaluar calidad de datos
    EvaluateDataQuality().process_rows(
        frame=dynamo_dyf,
        ruleset=DEFAULT_DATA_QUALITY_RULESET, 
        publishing_options={
            "dataQualityEvaluationContext": "EvaluateDataQuality_DynamoDB", 
            "enableDataQualityResultsPublishing": True
        }, 
        additional_options={
            "dataQualityResultsPublishing.strategy": "BEST_EFFORT", 
            "observations.scope": "ALL"
        }
    )

try:
    # 1. Leer datos de DynamoDB en Virginia (us-east-1)
    print(f"Leyendo datos desde DynamoDB en región {dynamo_region} (Virginia)...")

    dynamo_dyf = modulo_leer_datos_dynamo()
    
    evaluarCalidadDatos(dynamo_dyf)

    # Convertir a DataFrame
    dynamo_df = dynamo_dyf.toDF()
    total_dynamo_records = dynamo_df.count()
    print(f"Registros en DynamoDB: {total_dynamo_records}")
    
    # Si no hay registros en DynamoDB, terminar el proceso
    if total_dynamo_records == 0:
        print("No hay registros en DynamoDB. Finalizando el proceso.")
        job.commit()
        sys.exit(0)
    
    # 2. Configurar explícitamente el cliente de S3 para usar Ohio (us-east-2)
    print(f"Configurando acceso a S3 en región {s3_region} (Ohio)")
    spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", f"s3.{s3_region}.amazonaws.com")
    spark._jsc.hadoopConfiguration().set("fs.s3a.region", s3_region)
    
    # Intentar leer los datos existentes en S3 en Ohio (us-east-2)
    try:
        print(f"Leyendo datos existentes desde S3 en región {s3_region} (Ohio)...")
        # Verificar que la configuración de región esté correcta
        s3_client = get_s3_client(s3_region)
        
        # Leer recursivamente todos los archivos Parquet en todas las subcarpetas
        existing_s3_df = spark.read.option("recursiveFileLookup", "true").parquet(s3_target_path)
        existing_records_count = existing_s3_df.count()
        print(f"Registros existentes en S3 (todas las subcarpetas): {existing_records_count}")
        
        # 3. Identificar registros nuevos (no existen en S3)
        print("Identificando registros nuevos...")
        new_records_df = dynamo_df.join(
            existing_s3_df,
            on=primary_key,
            how="left_anti"
        )
        
        new_records_count = new_records_df.count()
        print(f"Registros nuevos a escribir: {new_records_count}")
        
        # 4. Solo escribir si hay registros nuevos
        if new_records_count > 0:
            print(f"Escribiendo {new_records_count} registros nuevos en S3...")
            
            # Preparar para escritura - añadir columnas de fecha y timestamp
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            new_records_prepared = new_records_df \
                .withColumn("fecha_procesamiento", lit(current_date_str)) \
                .withColumn("timestamp_procesamiento", lit(timestamp_str))
            
            # Convertir a DynamicFrame
            new_records_dyf = DynamicFrame.fromDF(new_records_prepared, glueContext, "new_records")
            
            print(f"Configurando escritura de archivos individuales en S3 región {s3_region} (Ohio)")
            # IMPORTANTE: Usa la nueva función para escribir Parquet de manera segura
            write_by_pk_to_s3_parquet_safe(
                dynamic_frame=new_records_dyf,
                primary_key=primary_key,
                target_path=s3_new_records_path,
            )
            
            print(f"Escritura completada: {new_records_count} registros escritos en {s3_new_records_path} (región: {s3_region})")
        else:
            print("No hay nuevos registros para escribir. No se crea ninguna carpeta nueva.")
            
    except Exception as s3_error:
        print(f"Error al leer datos existentes de S3 o no existen datos previos: {str(s3_error)}")
        print("Escribiendo todos los registros de DynamoDB por primera vez...")
        
        # En este caso, todos los registros de DynamoDB son nuevos
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        all_records_prepared = dynamo_df \
            .withColumn("fecha_procesamiento", lit(current_date_str)) \
            .withColumn("timestamp_procesamiento", lit(timestamp_str))
        
        # Convertir a DynamicFrame
        all_records_dyf = DynamicFrame.fromDF(all_records_prepared, glueContext, "all_records")
        
        # Segunda parte: cuando no hay datos existentes previos
        print(f"Configurando escritura inicial de archivos individuales en S3 región {s3_region} (Ohio)")
        # IMPORTANTE: Usa la nueva función para escribir Parquet de manera segura
        write_by_pk_to_s3_parquet_safe(
            dynamic_frame=all_records_dyf,
            primary_key=primary_key,
            target_path=s3_new_records_path
        )
        
        print(f"Escritura inicial completada: {total_dynamo_records} registros escritos en {s3_new_records_path}")

except Exception as e:
    print(f"Error durante la ejecución del job: {str(e)}")
    raise e

job.commit()
