import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id='AKIAXYKJTWQXW3AF6MR6',
                      aws_secret_access_key='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')

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

uploaded = upload_to_aws('s3_data.csv', 'mybucketpaulcorellausfq', 's3_file_data.csv')
