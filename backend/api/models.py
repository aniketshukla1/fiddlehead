"""The billable object graph: users → orgs → projects → models/datasets,
plus api_keys and usage_events (the metering that revenue is computed from)."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from .db import Base


def uid() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=uid)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=now)
    email_verified = Column(Boolean, default=False)
    # two-factor auth (TOTP, RFC 6238)
    totp_secret = Column(String, nullable=True)          # base32; set at enrolment
    totp_enabled = Column(Boolean, default=False)
    recovery_codes = Column(JSON, nullable=True)         # sha256 hashes of one-time codes
    memberships = relationship("Membership", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """A signed-in device. The JWT carries this row's id as its `jti`, so a
    session can be revoked server-side (sign out here / everywhere)."""
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=uid)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    active_org_id = Column(String, nullable=True)        # which workspace this device is in
    user_agent = Column(String, nullable=True)
    ip = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)
    last_seen_at = Column(DateTime, default=now)
    revoked = Column(Boolean, default=False)


class AuditEvent(Base):
    """Security-relevant activity, surfaced in settings and kept for compliance."""
    __tablename__ = "audit_events"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, index=True, nullable=True)
    user_id = Column(String, index=True, nullable=True)
    actor = Column(String, nullable=True)                # name/email snapshot
    action = Column(String, nullable=False)              # e.g. auth.login, member.invite
    detail = Column(String, nullable=True)
    ip = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)


class Org(Base):
    __tablename__ = "orgs"
    id = Column(String, primary_key=True, default=uid)
    name = Column(String, nullable=False)
    plan = Column(String, default="free")        # free | pro | team
    personal = Column(Boolean, default=True)
    credits = Column(Float, default=5.0)         # compute credits (free grant)
    created_at = Column(DateTime, default=now)
    # billing (Stripe); empty until the workspace subscribes
    stripe_customer_id = Column(String, nullable=True, index=True)
    stripe_subscription_id = Column(String, nullable=True)
    plan_status = Column(String, nullable=True)   # active | past_due | canceled
    plan_renews_at = Column(DateTime, nullable=True)
    members = relationship("Membership", back_populates="org", cascade="all, delete-orphan")


class OAuthAccount(Base):
    """Links a social login to a Fiddlehead user."""
    __tablename__ = "oauth_accounts"
    id = Column(String, primary_key=True, default=uid)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    provider = Column(String, index=True, nullable=False)      # google | github
    provider_user_id = Column(String, index=True, nullable=False)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)


class Membership(Base):
    __tablename__ = "memberships"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    role = Column(String, default="owner")       # owner | admin | member
    org = relationship("Org", back_populates="members")
    user = relationship("User", back_populates="memberships")


class Invite(Base):
    """A pending workspace invitation for an email that has no account yet.
    Consumed at signup; existing users are added to the org directly."""
    __tablename__ = "invites"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    email = Column(String, index=True, nullable=False)
    role = Column(String, default="member")      # admin | member
    invited_by = Column(String, nullable=True)   # user id
    accepted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=now)


class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    name = Column(String, nullable=False)
    template = Column(String, default="text")
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)


class Model(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    name = Column(String, nullable=False)
    kind = Column(String, default="scratch")     # scratch | finetune | asr | vlm | tts
    modality = Column(String, default="text")
    base_model = Column(String, nullable=True)
    params_m = Column(Float, nullable=True)
    path = Column(String, nullable=True)         # on-disk (or object-storage) artifact
    metrics = Column(JSON, nullable=True)
    deployed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=now)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    name = Column(String, nullable=False)
    stats = Column(JSON, nullable=True)
    path = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    name = Column(String, default="default")
    prefix = Column(String, index=True)          # shown in the UI; full key never stored
    key_hash = Column(String, nullable=False)    # sha256 of the full key
    created_at = Column(DateTime, default=now)
    last_used_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    model_name = Column(String, nullable=False)
    target = Column(String, default="local")      # local | cloud | cluster
    gpu = Column(String, nullable=True)
    gpus_per_node = Column(Integer, default=1)
    nodes = Column(Integer, default=1)
    strategy = Column(String, nullable=True)       # single | ddp | fsdp | tp+pp
    status = Column(String, default="queued")      # queued | running | done | failed
    gpu_seconds = Column(Float, default=0.0)        # metered → billing
    plan = Column(JSON, nullable=True)
    spec = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)


class UsageEvent(Base):
    __tablename__ = "usage_events"
    id = Column(String, primary_key=True, default=uid)
    org_id = Column(String, ForeignKey("orgs.id"), index=True, nullable=False)
    api_key_id = Column(String, ForeignKey("api_keys.id"), nullable=True)
    kind = Column(String)                        # train | infer
    units = Column(Float, default=1.0)           # gpu-seconds (train) or calls (infer)
    model_id = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)
