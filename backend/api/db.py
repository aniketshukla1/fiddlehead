"""Database engine + session (SQLAlchemy 2.0)."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_columns():
    """Additive dev migration: add columns the ORM declares but the existing
    table is missing, so upgrading in place doesn't drop anyone's data.
    (Production should run real Alembic migrations; this only ever ADDs.)"""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    added = []
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue                                   # create_all handles new tables
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in have:
                continue
            try:
                sql_type = col.type.compile(engine.dialect)
            except Exception:
                continue
            ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {sql_type}"
            default = getattr(col.default, "arg", None)
            if default is not None and not callable(default):
                literal = ("1" if default else "0") if isinstance(default, bool) else repr(default)
                ddl += f" DEFAULT {literal}"
            with engine.begin() as conn:
                conn.execute(text(ddl))
            added.append(f"{table.name}.{col.name}")
    return added
