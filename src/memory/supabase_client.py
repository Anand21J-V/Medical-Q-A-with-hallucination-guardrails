"""
src/memory/supabase_client.py
Singleton Supabase client using the Python SDK.
"""
from functools import lru_cache
from supabase import create_client, Client
from src.utils.config import get_settings

@lru_cache(maxsize=1)
def get_supabase() -> Client:
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)