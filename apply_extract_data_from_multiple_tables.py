import boto3
import time
import json
import logging
from botocore.exceptions import ClientError

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar cliente de Glue
glue_client = boto3.client('glue')

# Nombre del trabajo de Glue
job_name = 'generic_extract_table_information_from_dynamo_to_s3'

database = 'general'
bucket = 'mf-gluesparkscripts'

# Lista de configuraciones (cada elemento es un conjunto de parámetros para una ejecución)
job_configs = [
    #1. bd_users_table - users
    {
        '--bucket_name': bucket,
        '--catalog_database': database,
        '--catalog_table_name': 'users',
        '--dynamo_table_name': 'ERC_USERS_TABLE',
        '--folder_name': 'users',
        '--primary_key': 'id'
    },
    
    # 2. bd-contracts - contracts
    {
        '--bucket_name': bucket,
        '--catalog_database': database,
        '--catalog_table_name': 'contracts',
        '--dynamo_table_name': 'ERC_Contracts',
        '--folder_name': 'contracts',
        '--primary_key': 'idToken'
    },
    
    #3. erc_utilidades - erc_utilidades
    {
        '--bucket_name': bucket,
        '--catalog_database': database,
        '--catalog_table_name': 'erc_utilidades',
        '--dynamo_table_name': 'ERC_utilidades',
        '--folder_name': 'ERC_Utilidades',
        '--primary_key': 'factoryAddress'
    },
    
    # 4. user_transactions_target - user_transaction
    {
        '--bucket_name': bucket,
        '--catalog_database': database,
        '--catalog_table_name': 'user_transaction',
        '--dynamo_table_name': 'users_transactions',
        '--folder_name': 'user-transactions',
        '--primary_key': 'id'
    },
    {
        '--bucket_name': bucket,
        '--catalog_database': database,
        '--catalog_table_name': 'ERC_COUNTRY_CONTRACTS',
        '--dynamo_table_name': 'ERC_COUNTRY_CONTRACTS',
        '--folder_name': 'ERC_COUNTRY_CONTRACTS',
        '--primary_key': 'countryAddress'
    },
        # 5.  ERC Balances
    {
        '--bucket_name': bucket,
        '--catalog_database': database,
        '--catalog_table_name': 'erc_balances_table',
        '--dynamo_table_name': 'ERC_Balances',
        '--folder_name': 'ERC_Balances',
        '--primary_key': 'walletAddress'
    }
]

# Tiempo máximo de espera para intentar iniciar un job (30 minutos)
MAX_WAIT_TIME_SECONDS = 1800
# Tiempo entre intentos (30 segundos)
RETRY_INTERVAL_SECONDS = 30
# Número máximo de reintentos para un job fallido
MAX_RETRIES = 3

def check_job_status(job_name, run_id):
    """Verifica el estado de ejecución de un trabajo de Glue"""
    while True:
        try:
            response = glue_client.get_job_run(JobName=job_name, RunId=run_id)
            status = response['JobRun']['JobRunState']
            
            logger.info(f"Estado actual del trabajo: {status}")
            
            if status in ['SUCCEEDED', 'FAILED', 'TIMEOUT', 'STOPPED', 'ERROR']:
                return status
            
            # Esperar 30 segundos antes de volver a verificar
            time.sleep(RETRY_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Error al verificar el estado del trabajo: {str(e)}")
            # Si hay un error al verificar, esperamos un poco y volvemos a intentarlo
            time.sleep(RETRY_INTERVAL_SECONDS)

def start_job_with_retry(job_name, config):
    """Intenta iniciar un trabajo de Glue con reintentos si hay errores de concurrencia"""
    start_time = time.time()
    
    while (time.time() - start_time) < MAX_WAIT_TIME_SECONDS:
        try:
            logger.info(f"Intentando iniciar el trabajo con configuración: {json.dumps(config, indent=2)}")
            response = glue_client.start_job_run(
                JobName=job_name,
                Arguments=config
            )
            return response['JobRunId']
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            # Si el error es de concurrencia, esperamos y reintentamos
            if error_code == 'ConcurrentRunsExceededException':
                logger.warning("Se excedió el límite de ejecuciones concurrentes. Esperando para reintentar...")
                time.sleep(RETRY_INTERVAL_SECONDS)
            else:
                # Para otros errores, los propagamos
                logger.error(f"Error al iniciar el trabajo: {str(e)}")
                raise
    
    # Si llegamos aquí, hemos esperado el tiempo máximo sin éxito
    raise Exception(f"No se pudo iniciar el trabajo después de esperar {MAX_WAIT_TIME_SECONDS} segundos")

def run_job_with_configs():
    """Ejecuta el trabajo de Glue con cada configuración secuencialmente"""
    failed_jobs = []
    
    for i, config in enumerate(job_configs):
        logger.info(f"\n--- Iniciando ejecución {i+1}/{len(job_configs)} ---")
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Intentar iniciar el trabajo (con reintentos si hay errores de concurrencia)
                run_id = start_job_with_retry(job_name, config)
                logger.info(f"Trabajo iniciado. Run ID: {run_id}")
                
                # Esperar a que el trabajo se complete
                final_status = check_job_status(job_name, run_id)
                
                logger.info(f"Ejecución {i+1} completada con estado: {final_status}")
                
                # Si el trabajo fue exitoso, pasamos al siguiente
                if final_status == 'SUCCEEDED':
                    break
                
                # Si falló, incrementamos el contador de reintentos
                logger.warning(f"El trabajo falló. Reintento {retry_count + 1}/{MAX_RETRIES}")
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Error inesperado: {str(e)}")
                retry_count += 1
                
                if retry_count < MAX_RETRIES:
                    logger.info(f"Reintentando ({retry_count}/{MAX_RETRIES})...")
                    time.sleep(RETRY_INTERVAL_SECONDS)
        
        # Si agotamos los reintentos y el trabajo sigue fallando, lo registramos
        if retry_count >= MAX_RETRIES:
            failed_job = {
                'config': config,
                'index': i
            }
            failed_jobs.append(failed_job)
            logger.error(f"No se pudo completar la ejecución {i+1} después de {MAX_RETRIES} intentos")
    
    # Resumen de trabajos fallidos
    if failed_jobs:
        logger.error(f"\n--- Resumen de trabajos fallidos ({len(failed_jobs)}/{len(job_configs)}) ---")
        for job in failed_jobs:
            logger.error(f"Trabajo {job['index'] + 1}: {json.dumps(job['config'], indent=2)}")
    else:
        logger.info("\n--- Todos los trabajos se completaron correctamente ---")

if __name__ == "__main__":
    logger.info(f"Iniciando ejecuciones secuenciales del trabajo: {job_name}")
    logger.info(f"Se ejecutarán {len(job_configs)} configuraciones diferentes")
    
    run_job_with_configs()
    
    logger.info("\nProceso completado.")
