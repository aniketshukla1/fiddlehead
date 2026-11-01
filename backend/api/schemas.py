"""Pydantic v2 request/response schemas."""
from typing import Any, Optional

from pydantic import BaseModel, EmailStr


class ORMModel(BaseModel):
    model_config = {"from_attributes": True}


# ---- auth ----
class SignupIn(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(ORMModel):
    id: str
    name: str
    email: EmailStr
    email_verified: bool = False


class VerifyIn(BaseModel):
    token: str


class OrgOut(ORMModel):
    id: str
    name: str
    plan: str
    credits: float


class AuthOut(BaseModel):
    token: str
    user: UserOut
    org: OrgOut


class LoginOut(BaseModel):
    """Either a real session, or a challenge when 2FA is on."""
    mfa_required: bool = False
    mfa_token: Optional[str] = None
    token: Optional[str] = None
    user: Optional[UserOut] = None
    org: Optional[OrgOut] = None


class MfaIn(BaseModel):
    mfa_token: str
    code: str


# ---- sessions / 2FA / audit / workspaces ----
class SessionOut(ORMModel):
    id: str
    user_agent: Optional[str] = None
    ip: Optional[str] = None
    created_at: Any
    last_seen_at: Optional[Any] = None
    current: bool = False


class TotpSetupOut(BaseModel):
    secret: str
    uri: str


class TotpEnableIn(BaseModel):
    code: str


class TotpEnabledOut(BaseModel):
    recovery_codes: list[str]


class TotpDisableIn(BaseModel):
    password: str


class AuditOut(ORMModel):
    id: str
    action: str
    actor: Optional[str] = None
    detail: Optional[str] = None
    ip: Optional[str] = None
    created_at: Any


class WorkspaceOut(BaseModel):
    id: str
    name: str
    plan: str
    role: str
    personal: bool
    members: int
    current: bool = False


# ---- account & settings ----
class ProfileIn(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class PasswordIn(BaseModel):
    current_password: str
    new_password: str


class OrgIn(BaseModel):
    name: str


class MemberOut(BaseModel):
    id: str                      # membership id
    user_id: str
    name: str
    email: EmailStr
    role: str
    is_you: bool = False


class InviteIn(BaseModel):
    email: EmailStr
    role: str = "member"


class InviteOut(ORMModel):
    id: str
    email: EmailStr
    role: str
    created_at: Any


class RoleIn(BaseModel):
    role: str


class AccountOut(BaseModel):
    """Everything the settings area needs in one call."""
    user: UserOut
    org: OrgOut
    role: str                    # the caller's role in the org
    member_count: int
    model_count: int
    key_count: int
    created_at: Any              # user created_at
    org_created_at: Any
    org_personal: bool
    two_factor: bool = False
    session_count: int = 1
    workspace_count: int = 1


class DeleteAccountIn(BaseModel):
    password: str
    confirm: str                 # must equal the user's email


# ---- projects ----
class ProjectIn(BaseModel):
    name: str
    template: str = "text"


class ProjectOut(ORMModel):
    id: str
    name: str
    template: str
    created_at: Any


# ---- models ----
class ModelIn(BaseModel):
    name: str
    kind: str = "scratch"
    modality: str = "text"
    base_model: Optional[str] = None
    params_m: Optional[float] = None
    path: Optional[str] = None
    metrics: Optional[dict] = None
    project_id: Optional[str] = None


class ModelOut(ORMModel):
    id: str
    name: str
    kind: str
    modality: str
    base_model: Optional[str] = None
    params_m: Optional[float] = None
    metrics: Optional[dict] = None
    deployed: bool
    created_at: Any


# ---- api keys ----
class ApiKeyIn(BaseModel):
    name: str = "default"


class ApiKeyOut(ORMModel):
    id: str
    name: str
    prefix: str
    created_at: Any
    last_used_at: Optional[Any] = None
    revoked: bool


class ApiKeyCreated(ApiKeyOut):
    key: str  # the full key, shown exactly once


# ---- usage ----
class UsageOut(ORMModel):
    id: str
    kind: str
    units: float
    model_id: Optional[str] = None
    created_at: Any


class UsageSummary(BaseModel):
    train_units: float
    infer_units: float
    credits: float
    events: list[UsageOut]
