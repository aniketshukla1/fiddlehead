# Security policy

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Report them privately through GitHub:
**Security → Report a vulnerability** on this repository. You'll get an acknowledgement within a few days and
updates as the fix progresses; credit is given in the release notes unless you'd rather stay anonymous.

Helpful details: what an attacker could do, the steps to reproduce, and the version or commit you tested.

## Supported versions

Fixes land on `main` and in the next release. Older releases aren't patched.

## Running Fiddlehead safely

- **Secrets.** Copy `backend/.env.example` to `backend/.env` for real deployments. Never commit `.env` files —
  they're git-ignored. If `JWT_SECRET` is unset, a random one is generated on first run and kept in
  `backend/.jwt_secret`; set it explicitly when running more than one API process.
- **Network exposure.** The training engine (`ws://localhost:8765`) has no authentication: it trains whatever
  it's sent and reads paths and URLs given as data sources. Keep it on `localhost` or a private network, and only
  expose the API, behind HTTPS.
- **Accounts.** The API hashes passwords with bcrypt, keeps sessions server-side so they can be revoked, supports
  two-factor authentication, and rate-limits sign-in. Run `python backend/check_config.py` to see what's configured.
