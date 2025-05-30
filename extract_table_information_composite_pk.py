import sys
import datetime
import boto3
import uuid
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from awsgluedq.transforms import EvaluateDataQuality
from pyspark.sql.functions import lit, current_date, current_timestamp, col, hash, struct, concat_ws
from pyspark.sql.types import StringType, StructType, StructField, MapType
import pyarrow

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
    'primary_key',                # Clave primaria de la tabla (separada por comas para composite)
    'catalog_database',           # Base de datos del catálogo
    'catalog_table_name',         # Nombre de la tabla en el catálogo
    'primary_key_is_composite'    # Debe ser 'true' para este trabajo
]

# Obtenemos los parámetros requeridos
args = getResolvedOptions(sys.argv, required_params)

# Inicializar el contexto y job de Glue
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Obtener valores de los parámetros requeridos
dynamo_table_name = args['dynamo_table_name']
bucket_name = args['bucket_name']
folder_name = args['folder_name']
catalog_database = args['catalog_database']
catalog_table_name = args['catalog_table_name']
primary_key = args['primary_key']

# Verificar que primary_key_is_composite sea true
if args['primary_key_is_composite'].lower() != 'true':
    print("ADVERTENCIA: Este trabajo está diseñado para claves primarias compuestas. El parámetro 'primary_key_is_composite' debe ser 'true'.")
    print("Continuando de todos modos asumiendo que la clave primaria es compuesta.")

# Separar las claves compuestas
composite_keys = primary_key.split(',')
print(f"Claves primarias compuestas: {composite_keys}")

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
print(f"Ruta base en S3: {s3_target_path}")
print(f"Ruta para nuevos registros: {s3_new_records_path}")


from pyspark.sql.types import _parse_datatype_string

def infer_unified_schema_with_composite_keys(df, composite_keys):
    """
    Infiere un esquema unificado que:
    1. Captura todos los campos encontrados en los datos
    2. Respeta los tipos de datos originales cuando es posible
    3. Asegura que las claves compuestas tengan el tipo correcto
    
    Args:
        df: DataFrame de Spark con los datos
        composite_keys: Lista de nombres de columnas que forman la PK compuesta
        
    Returns:
        StructType: Esquema unificado con todos los campos
    """
    # Paso 1: Recolectar todos los nombres de campos
    all_fields = set(df.columns)
    
    # Paso 2: Analizar una muestra para detectar campos anidados
    sample_size = min(1000, df.count())
    sample_data = df.limit(sample_size).collect()
    
    for row in sample_data:
        for field_name in row.__fields__:
            if field_name not in all_fields:
                all_fields.add(field_name)
    
    print(f"Campos detectados: {len(all_fields)} (incluyendo {len(composite_keys)} claves primarias)")
    
    # Paso 3: Inferir tipos para cada campo
    schema_fields = []
    type_samples = {}
    
    # Primero procesamos las claves primarias para asegurar su tipo
    for pk in composite_keys:
        if pk not in all_fields:
            raise ValueError(f"Clave primaria '{pk}' no encontrada en los datos")
        
        # Muestrear valores para inferir tipo
        pk_samples = df.select(pk).limit(100).collect()
        pk_type = _infer_type_from_samples([row[pk] for row in pk_samples])
        
        print(f"Tipo inferido para clave primaria '{pk}': {pk_type}")
        schema_fields.append(StructField(pk, pk_type, False))  # Claves no nulas
        type_samples[pk] = pk_type
    
    # Luego procesamos el resto de los campos
    for field in sorted(all_fields - set(composite_keys)):
        # Si ya procesamos este campo como PK, lo saltamos
        if field in composite_keys:
            continue
            
        # Muestrear valores para inferir tipo
        field_samples = df.select(field).limit(100).collect()
        field_type = _infer_type_from_samples([row[field] for row in field_samples])
        
        schema_fields.append(StructField(field, field_type, True))  # Campos normales pueden ser nulos
        type_samples[field] = field_type
    
    unified_schema = StructType(schema_fields)
    
    # Validación final
    _validate_schema_with_composite_keys(unified_schema, composite_keys)
    
    return unified_schema

def _infer_type_from_samples(samples):
    """
    Infiere el tipo de datos Spark apropiado a partir de una muestra de valores.
    """
    from pyspark.sql.types import (StringType, IntegerType, LongType, 
                                  DoubleType, BooleanType, TimestampType,
                                  DateType, BinaryType)
    
    # Filtramos valores nulos
    non_null_samples = [x for x in samples if x is not None]
    
    if not non_null_samples:
        return StringType()  # Default para columnas completamente nulas
    
    # Chequear tipos específicos en orden de preferencia
    sample = non_null_samples[0]
    
    # 1. Boolean
    if isinstance(sample, bool):
        if all(isinstance(x, bool) or x is None for x in samples):
            return BooleanType()
    
    # 2. Números enteros
    try:
        if all(isinstance(x, int) and not isinstance(x, bool) or x is None for x in samples):
            if all(-2147483648 <= x <= 2147483647 for x in non_null_samples):
                return IntegerType()
            return LongType()
    except:
        pass
    
    # 3. Números decimales
    try:
        if all(isinstance(x, (int, float)) or x is None for x in samples):
            return DoubleType()
    except:
        pass
    
    # 4. Fechas y timestamps (manejo especial para strings)
    if isinstance(sample, str):
        try:
            from dateutil.parser import parse
            if len(sample) == 10 and sample[4] == '-' and sample[7] == '-':
                if all(parse(x).date() if x else True for x in non_null_samples):
                    return DateType()
            else:
                if all(parse(x) if x else True for x in non_null_samples):
                    return TimestampType()
        except:
            pass
    
    # 5. Binarios
    if isinstance(sample, (bytes, bytearray)):
        return BinaryType()
    
    # Default a String
    return StringType()

def _validate_schema_with_composite_keys(schema, composite_keys):
    """
    Valida que el esquema sea compatible con las claves primarias compuestas.
    """
    schema_fields = {field.name: field for field in schema.fields}
    
    for pk in composite_keys:
        if pk not in schema_fields:
            raise ValueError(f"Clave primaria '{pk}' no encontrada en el esquema")
        
        if schema_fields[pk].nullable:
            raise ValueError(f"Clave primaria '{pk}' no puede ser nullable")

def enforce_schema_with_composite_keys(df, unified_schema, composite_keys):
    """
    Aplica el esquema unificado al DataFrame, asegurando:
    1. Todas las columnas están presentes
    2. Las claves primarias no son nulas
    3. Los tipos de datos son correctos
    """
    # Paso 1: Asegurar que todas las columnas existan
    existing_cols = set(df.columns)
    
    for field in unified_schema:
        if field.name not in existing_cols:
            df = df.withColumn(field.name, lit(None).cast(field.dataType))
    
    # Paso 2: Convertir tipos manteniendo las claves primarias
    for field in unified_schema:
        if field.name in composite_keys:
            # Para claves primarias, usar coalesce para evitar nulos
            df = df.withColumn(
                field.name, 
                coalesce(col(field.name).cast(field.dataType), 
                         lit("NULL_" + field.name))  # Valor por defecto seguro
            )
        else:
            df = df.withColumn(field.name, col(field.name).cast(field.dataType))
    
    # Paso 3: Seleccionar solo las columnas del esquema en el orden correcto
    return df.select([field.name for field in unified_schema])


def write_consistent_parquet_files(df, primary_key, target_path, is_composite=False):
    """
    Versión unificada para PK simples y compuestas que conserva todos los caracteres
    """
    # Configuración S3
    s3_client = boto3.client('s3', region_name=s3_region)
    bucket = target_path.replace("s3://", "").split("/")[0]
    prefix = target_path.replace(f"s3://{bucket}/", "")
    
    if is_composite:
        # Procesamiento para PK compuesta
        if isinstance(primary_key, str):
            composite_keys = primary_key.split(',')
        else:
            composite_keys = primary_key
        
        print(f"Procesando {df.count()} registros con claves compuestas: {composite_keys}")
        
        # Obtener combinaciones únicas de PKs
        pk_combinations = df.select(composite_keys).distinct().collect()
        print(f"Encontradas {len(pk_combinations)} combinaciones únicas de PK")
        
        for combo in pk_combinations:
            try:
                # Construir filtro para esta combinación de PKs
                filter_cond = None
                pk_values = []
                
                for key in composite_keys:
                    val = combo[key]
                    pk_values.append(str(val) if val is not None else "NULL")
                    
                    if filter_cond is None:
                        filter_cond = (col(key) == val)
                    else:
                        filter_cond = filter_cond & (col(key) == val)
                
                # Generar nombre de archivo exacto con los valores PK
                pk_filename = "_".join(pk_values)
                # Reemplazar solo caracteres problemáticos para S3
                pk_filename = pk_filename.replace(" ", "_").replace("\\", "_").replace(":", "_")
                file_name = f"{pk_filename}.parquet"
                
                # Filtrar los registros con esta PK
                filtered_df = df.filter(filter_cond)
                print(f"Escribiendo {filtered_df.count()} registros para PK: {pk_values}")
                
                # Escribir temporalmente
                temp_path = f"s3://{bucket}/temp_{uuid.uuid4()}/"
                filtered_df.write.mode("overwrite").parquet(temp_path)
                
                # Mover a ubicación final
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=temp_path.replace(f"s3://{bucket}/", "")
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if obj['Key'].endswith('.parquet'):
                            s3_client.copy_object(
                                Bucket=bucket,
                                CopySource={'Bucket': bucket, 'Key': obj['Key']},
                                Key=f"{prefix}{file_name}"
                            )
                            s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
                            print(f"Archivo movido a: {prefix}{file_name}")
                
            except Exception as e:
                print(f"Error procesando combinación {pk_values}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise e
    else:
        # Procesamiento para PK simple
        for row in df.collect():
            
            pk_value = str(row[primary_key]) if row[primary_key] is not None else f"null_{uuid.uuid4().hex}"
            file_name = f"{pk_value}.parquet"  # Nombre exacto con la PK
            single_row_df = spark.createDataFrame([row.asDict()], df.schema)
            _write_single_parquet_to_s3(single_row_df, s3_client, bucket, prefix, file_name)
                

    print(f"Escritura completada en s3://{bucket}/{prefix}")

def _write_single_parquet_to_s3(df, s3_client, bucket, prefix, file_name):
    """Función auxiliar para escribir un Parquet individual en S3"""
    # Escribir temporalmente
    temp_path = f"s3://{bucket}/temp_{uuid.uuid4()}/"
    df.write.parquet(temp_path)
    
    # Mover a ubicación final
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=temp_path.replace(f"s3://{bucket}/", "")
    )
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.parquet'):
                s3_client.copy_object(
                    Bucket=bucket,
                    CopySource={'Bucket': bucket, 'Key': obj['Key']},
                    Key=f"{prefix}{file_name}"
                )
                s3_client.delete_object(Bucket=bucket, Key=obj['Key'])

def modulo_leer_datos_dynamo():
    """
    Lee datos de una tabla DynamoDB y los convierte a un formato adecuado para Spark.
    Maneja todos los tipos de datos nativos de DynamoDB y asegura que todos los campos
    se procesen correctamente.
    """
    print(f"Conectando a DynamoDB en región {dynamo_region}...")


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
    
    # Para clave compuesta, usamos todas las columnas de clave primaria para el join
    new_records_df = dynamo_df.join(
        existing_s3_df,
        on=composite_keys,
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
        # Llamada a la función especializada para claves primarias compuestas
        write_consistent_parquet_files(
            df=new_records_prepared,  # Usamos el DataFrame directamente
            primary_key=composite_keys,  # Lista de claves compuestas
            target_path=s3_new_records_path,
            is_composite=True
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
    # Llamada a la función especializada para claves primarias compuestas
    write_consistent_parquet_files(
        df=dynamo_df,  # Usamos el DataFrame directamente
        primary_key=composite_keys,  # Lista de claves compuestas
        target_path=s3_new_records_path,
        is_composite=True
    )

    
    print(f"Escritura inicial completada: {total_dynamo_records} registros escritos en {s3_new_records_path}")




job.commit()
