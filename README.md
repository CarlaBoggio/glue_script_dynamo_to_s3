# glue_script_dynamo_to_s3
Este script comparará datos entre DynamoDB y un bucket S3, identificando los registros que existen en DynamoDB pero no en S3, y escribiéndolos en una nueva carpeta con formato year/month/day/timestamp de manera incremental
