from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os


def _load_dotenv(override: bool = False, env_file: Path | None = None) -> None:
    if env_file is not None and env_file.is_file():
        candidate_paths = (env_file,)
    else:
        project_env = Path(__file__).resolve().parents[3] / ".env"
        cwd_env = Path.cwd() / ".env"
        candidate_paths = (project_env, cwd_env)

    for candidate in candidate_paths:
        if candidate.is_file():
            # 1. Prefer python-dotenv for complete spec compliance
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=candidate, override=override)
                return
            except Exception:
                pass
            # 2. Fallback parser if dotenv import fails
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and (override or k not in os.environ):
                        os.environ[k] = v
            except Exception:
                pass
            break


_load_dotenv(override=False)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(slots=True)
class AppSettings:
    app_data_dir: Path = field(default_factory=lambda: Path(_env("APP_DATA_DIR", "./data/runtime")))
    upload_dir: Path = field(default_factory=lambda: Path(_env("UPLOAD_DIR", "./data/uploads")))
    sqlite_path: Path = field(default_factory=lambda: Path(_env("SQLITE_PATH", "./data/runtime/rag_benchmark_lab.sqlite3")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    mongodb_uri: str = field(default_factory=lambda: _env("MONGODB_URI", "").strip())
    mongodb_database: str = field(default_factory=lambda: _env("MONGODB_DATABASE", "rag_benchmark_lab").strip())
    mongodb_users_collection: str = field(default_factory=lambda: _env("MONGODB_USERS_COLLECTION", "users").strip())
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: _env("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "change-me"))

    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "rag_chunks"))

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower().strip())
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", "").strip())
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile").strip())
    groq_timeout_seconds: int = field(default_factory=lambda: _env_int("GROQ_TIMEOUT_SECONDS", 60))

    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip())
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash").strip())
    gemini_timeout_seconds: int = field(default_factory=lambda: _env_int("GEMINI_TIMEOUT_SECONDS", 60))

    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"))
    ollama_timeout_seconds: int = field(default_factory=lambda: _env_int("OLLAMA_TIMEOUT_SECONDS", 120))
    ollama_retries: int = field(default_factory=lambda: _env_int("OLLAMA_RETRIES", 2))
    llm_mock_mode: bool = field(default_factory=lambda: _env_bool("LLM_MOCK_MODE", False))

    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    default_top_k: int = field(default_factory=lambda: _env_int("DEFAULT_TOP_K", 5))
    graph_max_hops: int = field(default_factory=lambda: _env_int("GRAPH_MAX_HOPS", 2))
    graph_max_results: int = field(default_factory=lambda: _env_int("GRAPH_MAX_RESULTS", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    evaluation_model: str = field(default_factory=lambda: _env("EVALUATION_MODEL", "llama3.1:8b-instruct-q4_K_M"))
    evaluation_timeout: int = field(default_factory=lambda: _env_int("EVALUATION_TIMEOUT", 120))
    upload_size_limit_mb: int = field(default_factory=lambda: _env_int("UPLOAD_SIZE_LIMIT_MB", 25))
    answer_temperature: float = field(default_factory=lambda: _env_float("ANSWER_TEMPERATURE", 0.0))

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")
        if self.default_top_k <= 0:
            raise ValueError("DEFAULT_TOP_K must be positive")
        if self.graph_max_hops < 0:
            raise ValueError("GRAPH_MAX_HOPS must be non-negative")
        if self.graph_max_results <= 0:
            raise ValueError("GRAPH_MAX_RESULTS must be positive")
        if self.rrf_k <= 0:
            raise ValueError("RRF_K must be positive")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive")
        if self.evaluation_timeout <= 0:
            raise ValueError("EVALUATION_TIMEOUT must be positive")
        if self.upload_size_limit_mb <= 0:
            raise ValueError("UPLOAD_SIZE_LIMIT_MB must be positive")
        if self.gemini_timeout_seconds <= 0:
            raise ValueError("GEMINI_TIMEOUT_SECONDS must be positive")
        if self.groq_timeout_seconds <= 0:
            raise ValueError("GROQ_TIMEOUT_SECONDS must be positive")
        if not self.log_level:
            raise ValueError("LOG_LEVEL must not be empty")

    def ensure_directories(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, override: bool = False, env_file: Path | None = None) -> "AppSettings":
        _load_dotenv(override=override, env_file=env_file)
        settings = cls()
        settings.validate()
        settings.ensure_directories()
        return settings

    def __repr__(self) -> str:
        d = self.to_dict()
        items = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"AppSettings({items})"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in (
            "neo4j_password",
            "qdrant_api_key",
            "groq_api_key",
            "gemini_api_key",
            "mongodb_uri",
        ):
            data[key] = "***" if data.get(key) else ""
        data["app_data_dir"] = str(self.app_data_dir)
        data["upload_dir"] = str(self.upload_dir)
        data["sqlite_path"] = str(self.sqlite_path)
        return data


def get_settings(override: bool = False, env_file: Path | None = None) -> AppSettings:
    return AppSettings.from_env(override=override, env_file=env_file)


