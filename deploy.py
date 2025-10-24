import os
import boto3

# Get environment variables
IMAGE_URI = os.environ['IMAGE_URI']
SAGEMAKER_ROLE_ARN = os.environ['SAGEMAKER_ROLE_ARN']
REGION = os.environ.get('AWS_REGION', 'us-east-1')
MODEL_NAME = os.environ.get('SAGEMAKER_MODEL_NAME', 'iris-model')
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'iris-endpoint')

print(f"🚀 Deploying image {IMAGE_URI} to SageMaker...")

# Initialize SageMaker client
sm = boto3.client('sagemaker', region_name=REGION)

# 1. Create or update model
try:
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={'Image': IMAGE_URI, 'Mode': 'SingleModel'},
        ExecutionRoleArn=SAGEMAKER_ROLE_ARN
    )
    print("✅ Model created.")
except sm.exceptions.ResourceInUse:
    print("🔁 Model already exists. Skipping creation.")

# 2. Create or update endpoint config
endpoint_config_name = f"{MODEL_NAME}-config"
try:
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': MODEL_NAME,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium',
        }]
    )
    print("✅ Endpoint config created.")
except sm.exceptions.ResourceInUse:
    print("🔁 Endpoint config already exists. Skipping creation.")

# 3. Create or update endpoint
try:
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=endpoint_config_name
    )
    print("🚀 Endpoint creation started.")
except sm.exceptions.ResourceInUse:
    print("🔁 Updating existing endpoint...")
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=endpoint_config_name
    )
    print("✅ Endpoint update initiated.")
