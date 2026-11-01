// Account-scoped resources on the control-plane API.
import { api } from "./api.js";

// public: which optional integrations this deployment has on (billing, email, oauth)
export const serverMeta = () => api("/meta");

export const listProjects = () => api("/projects");
export const createProject = (name, template) => api("/projects", { method: "POST", body: { name, template } });
export const deleteProject = (id) => api(`/projects/${id}`, { method: "DELETE" });

export const listModels = () => api("/models");
export const registerModel = (m) => api("/models", { method: "POST", body: m });
export const deployModel = (id) => api(`/models/${id}/deploy`, { method: "POST" });
export const undeployModel = (id) => api(`/models/${id}/undeploy`, { method: "POST" });
export const deleteModelApi = (id) => api(`/models/${id}`, { method: "DELETE" });

export const listKeys = () => api("/keys");
export const createKey = (name) => api("/keys", { method: "POST", body: { name } });
export const revokeKey = (id) => api(`/keys/${id}`, { method: "DELETE" });

export const getUsage = () => api("/usage");

// ---- account & workspace settings ----
export const getAccount = () => api("/account");
export const updateProfile = (body) => api("/account/profile", { method: "PATCH", body });
export const changePassword = (current_password, new_password) =>
  api("/account/password", { method: "POST", body: { current_password, new_password } });
export const renameOrg = (name) => api("/account/org", { method: "PATCH", body: { name } });

export const listMembers = () => api("/account/members");
export const setMemberRole = (id, role) => api(`/account/members/${id}`, { method: "PATCH", body: { role } });
export const removeMember = (id) => api(`/account/members/${id}`, { method: "DELETE" });

export const listInvites = () => api("/account/invites");
export const inviteMember = (email, role) => api("/account/invites", { method: "POST", body: { email, role } });
export const revokeInvite = (id) => api(`/account/invites/${id}`, { method: "DELETE" });

export const deleteAccount = (password, confirm) =>
  api("/account", { method: "DELETE", body: { password, confirm } });

// ---- signed-in devices ----
export const listSessions = () => api("/account/sessions");
export const revokeSession = (id) => api(`/account/sessions/${id}`, { method: "DELETE" });
export const revokeOtherSessions = () => api("/account/sessions/revoke-others", { method: "POST" });

// ---- two-factor auth ----
export const startTotp = () => api("/account/2fa/setup", { method: "POST" });
export const enableTotp = (code) => api("/account/2fa/enable", { method: "POST", body: { code } });
export const disableTotp = (password) => api("/account/2fa/disable", { method: "POST", body: { password } });

// ---- email verification ----
export const verifyEmail = (token) => api("/auth/verify-email", { method: "POST", body: { token } });
export const resendVerification = () => api("/auth/resend-verification", { method: "POST" });

// ---- social sign-in ----
export const oauthProviders = () => api("/auth/oauth/providers");
export const oauthStart = (provider) => api(`/auth/oauth/${provider}/start`);

// ---- billing ----
export const billingStatus = () => api("/billing/status");
export const startCheckout = (plan) => api("/billing/checkout", { method: "POST", body: { plan } });
export const billingPortal = () => api("/billing/portal", { method: "POST" });

// ---- distributed compute & training jobs ----
export const listGpus = () => api("/compute/gpus");
export const planCompute = (body) => api("/compute/plan", { method: "POST", body });
export const submitJob = (body) => api("/jobs", { method: "POST", body });
export const listJobs = () => api("/jobs");
export const getJob = (id) => api(`/jobs/${id}`);
export const cancelJob = (id) => api(`/jobs/${id}/cancel`, { method: "POST" });

// ---- workspaces & audit ----
export const listWorkspaces = () => api("/account/workspaces");
export const switchWorkspace = (id) => api(`/account/workspaces/${id}/switch`, { method: "POST" });
export const listAudit = () => api("/account/audit");
