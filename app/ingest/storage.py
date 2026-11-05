import asyncio
from functools import lru_cache

import boto3

from app.core import config


@lru_cache(maxsize=1)
def _build_client(endpoint_url: str | None, access_key: str | None, secret_key: str | None, region: str | None):
    # 운영은 인스턴스 역할을 쓰고, 로컬 MinIO만 주소와 키를 지정한다
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def _read(storage_key: str) -> bytes:
    client = _build_client(
        config.storage_endpoint_url(),
        config.storage_access_key(),
        config.storage_secret_key(),
        config.storage_region(),
    )
    response = client.get_object(Bucket=config.storage_bucket(), Key=config.storage_prefix() + storage_key.lstrip("/"))
    with response["Body"] as body:
        return body.read()


async def read_object(storage_key: str) -> bytes:
    return await asyncio.to_thread(_read, storage_key)
