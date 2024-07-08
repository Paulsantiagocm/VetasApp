import boto3
s3_client = boto3.client('s3')

response = s3_client.create_bucket(
	ACL = 'private',
    Bucket = 'mybucketpaulcorellausfq'
)

print(response)