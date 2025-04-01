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
primary_key = args['primary_key']
catalog_database = args['catalog_database']
catalog_table_name = args['catalog_table_name']

# Obtener región actual de ejecución de Glue
current_region = boto3.session.Session().region_name
dynamo_account_id = get_aws_account_id()

# Definir parámetros opcionales con valores predeterminados
# Ya no es necesario pasarlos como argumentos al job
dynamo_region = "us-east-1"  # Virginia para DynamoDB
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

try:
    # 1. Leer datos de DynamoDB en Virginia (us-east-1)
    print(f"Leyendo datos desde DynamoDB en región {dynamo_region} (Virginia)...")
    dynamo_dyf = glueContext.create_dynamic_frame.from_options(
        connection_type="dynamodb",
        connection_options={
            "dynamodb.export": "ddb",
            "dynamodb.s3.bucket": s3_temp_bucket,
            "dynamodb.s3.prefix": s3_temp_prefix,
            "dynamodb.tableArn": dynamo_db_table_arn,
            "dynamodb.unnestDDBJson": True,
            "dynamodb.region": dynamo_region
        },
        transformation_ctx="dynamo_dyf"
    )

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
            
            # Configurar sink de S3 en Ohio (us-east-2)
            print(f"Configurando escritura en S3 región {s3_region} (Ohio)")
            s3_sink = glueContext.getSink(
                path=s3_new_records_path,
                connection_type="s3",
                updateBehavior="UPDATE_IN_DATABASE",
                partitionKeys=[],
                enableUpdateCatalog=True,
                transformation_ctx="s3_sink"
            )
            s3_sink.setCatalogInfo(
                catalogDatabase=catalog_database,
                catalogTableName=catalog_table_name
            )
            s3_sink.setFormat("glueparquet", compression="snappy")
            
            # Escribir registros
            s3_sink.writeFrame(new_records_dyf)
            
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
        
        # Configurar sink de S3 en Ohio (us-east-2)
        print(f"Configurando escritura inicial en S3 región {s3_region} (Ohio)")
        s3_sink = glueContext.getSink(
            path=s3_new_records_path,
            connection_type="s3",
            updateBehavior="UPDATE_IN_DATABASE",
            partitionKeys=[],
            enableUpdateCatalog=True,
            transformation_ctx="s3_sink"
        )
        s3_sink.setCatalogInfo(
            catalogDatabase=catalog_database,
            catalogTableName=catalog_table_name
        )
        s3_sink.setFormat("glueparquet", compression="snappy")
        
        # Escribir registros
        s3_sink.writeFrame(all_records_dyf)
        
        print(f"Escritura inicial completada: {total_dynamo_records} registros escritos en {s3_new_records_path}")

except Exception as e:
    print(f"Error durante la ejecución del job: {str(e)}")
    raise e

job.commit()
