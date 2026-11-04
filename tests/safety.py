"""테스트가 실제 API 키와 외부 네트워크를 쓰지 못하게 막는다. app을 import하기 전에 install()을 호출해야 한다."""

import os
import socket

import dotenv

FAKE_ENV = {
    "APP_ENV": "local",
    "INTERNAL_SECRET": "test-internal-secret",
    "USE_SAMPLE_DATA": "",
    "JEV_API_KEY": "fake-jev-key",
    "JEV_MODEL": "jev-test",
    "EMBEDDING_API_KEY": "fake-embedding-key",
    "EMBEDDING_MODEL": "openai:fake-embedding",
    "EMBEDDING_DIMENSIONS": "1024",
    "LLM_API_KEY": "fake-llm-key",
    "LLM_MODEL": "fake:fake-llm",
    "LLM_FALLBACK_MODEL": "",
    "DATABASE_URL": "",
}
PROVIDER_ENV = ["TYPESAFE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", None, ""}

blocked_attempts: list[str] = []


def install() -> None:
    dotenv.load_dotenv = lambda *args, **kwargs: False
    for name in PROVIDER_ENV:
        os.environ.pop(name, None)
    os.environ.update(FAKE_ENV)

    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if isinstance(address, tuple) and address[0] not in LOCAL_HOSTS:
            blocked_attempts.append(str(address[0]))
            raise OSError(f"테스트 중 외부 접속 차단: {address[0]}")
        return real_connect(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in LOCAL_HOSTS:
            blocked_attempts.append(str(host))
            raise OSError(f"테스트 중 외부 DNS 조회 차단: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.getaddrinfo = getaddrinfo
