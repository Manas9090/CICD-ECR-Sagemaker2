import argparse
import boto3
import time
import sys
import botocore.exceptions

def main():
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker")
    parser.add_argument("--image-uri", required=True, help="ECR image URI with tag")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument("--model-name", required=True, help="Name of the SageMaker model")
    parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    parser.add_argument("--role-arn", required=True, help="IAM Role ARN for SageMaker execution")
    args = parser.parse_args()

    # 🩹 Fix: Trim any trailing spaces or newline characters from the Role ARN
    args.role_arn = args.role_arn.strip()

    print(f"🚀 Deploying image {args.image_uri} to SageMaker...")
    print(f"📍 Using model name: {args.model_name}")
    print(f"📍 Endpoint name: {args.endpoint_name}")
    print(f"📍 Role ARN: {args.role_arn}")

    sm = boto3.client("sagemaker", region_name=args.region)

    # Step 1: Delete old endpoint and model if they exist (optional cleanup)
    try:
        print("🧹 Checking for existing endpoint or model...")
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        print("🗑️ Existing endpoint deleted.")
        time.sleep(10)

        sm.delete_endpoint_config(EndpointConfigName=args.endpoint_name)
        print("🗑️ Existing endpoint config deleted.")
        time.sleep(5)

        sm.delete_model(ModelName=args.model_name)
        print("🗑️ Existing model deleted.")
        time.sleep(5)

    except botocore.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print("✅ No existing endpoint or model found, continuing...")
        else:
            print(f"⚠️ Warning during cleanup: {e}")

    # Step 2: Create SageMaker model
    try:
        print("📦 Creating SageMaker model...")
        sm.create_model(
            ModelName=args.model_name,
            PrimaryContainer={
                "Image": args.image_uri,
                "Mode": "SingleModel",
            },
            ExecutionRoleArn=args.role_arn,
        )
        print("✅ Model created successfully!")
    except botocore.exceptions.ClientError as e:
        print(f"❌ Failed to create model: {e}")
        sys.exit(1)

    # Step 3: Create Endpoint Configuration
    try:
        print("⚙️ Creating endpoint configuration...")
        sm.create_endpoint_config(
            EndpointConfigName=args.endpoint_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": args.model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                }
            ],
        )
        print("✅ Endpoint configuration created!")
    except botocore.exceptions.ClientError as e:
        print(f"❌ Failed to create endpoint config: {e}")
        sys.exit(1)

    # Step 4: Create Endpoint
    try:
        print("🚀 Deploying endpoint (this may take several minutes)...")
        sm.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=args.endpoint_name,
        )

        # Optional: Wait until endpoint is InService
        print("⏳ Waiting for endpoint to become InService...")
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=args.endpoint_name)
        print("✅ Endpoint deployed and InService!")

    except botocore.exceptions.ClientError as e:
        print(f"❌ Failed to create endpoint: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
