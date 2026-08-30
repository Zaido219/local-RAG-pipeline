import json
import redis
from rag_engine.core.interfaces import SessionMemoryInterface
from typing import List


class RedisSessionMemoryStore(SessionMemoryInterface):
    def __init__(self,redis_client: redis.Redis,ttl_seconds:int = 300, max_messages:int = 10):
        self.client = redis_client
        self.ttl = ttl_seconds
        self.max_messages = max_messages

    def _get_key(self, session_id:str) -> str:
        return f"session: {session_id}:messages"

    def add_message(self, session_id:str, role:str, content:str) -> None:
        key = self._get_key(session_id)
        payload = json.dumps({"role":role, "content":content})

        pipe = self.client.pipeline()
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, self.max_messages - 1)
        pipe.expire(key, self.ttl)
        pipe.execute()

    def get_session_history(self, session_id:str) -> List[dict[str, str]]:
        key = self._get_key(session_id)
        raw_messages = self.client.lrange(key, 0, -1)

        if not raw_messages:
            return []

        self.client.expire(key, self.ttl)
        messages = [json.loads(msg.decode("utf-8")) for msg in raw_messages]

        return list(reversed(messages))

    def clear_session(self, session_id:str) -> None:
        self.client.delete(self._get_key(session_id))