import logging
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_supabase_client = None


def get_supabase():
    """Lazy loader for Supabase client to avoid crashing on missing/placeholder credentials."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY or "your_" in SUPABASE_URL:
            logger.warning("Valid SUPABASE_URL or SUPABASE_KEY not found in environment variables")
            return None
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client
