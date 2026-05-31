# Auth

This document is the source of truth for authentication, Square OAuth onboarding, token lifecycle, and deployment requirements.

## What This Adds

- App-level user authentication with email/password.
- Session-based login using an HTTP-only cookie.
- Self-serve Square OAuth onboarding per app user.
- Encrypted storage for Square access and refresh tokens.
- Automatic Square access-token refresh during sync.
- User-scoped Square sync endpoints.

## User Flow

### New client onboarding

1. User registers in the app with email and password.
2. User logs in.
3. User clicks `Connect Square`.
4. Browser is redirected to Square.
5. User signs into their Square account and approves access.
6. Square redirects back to this app callback URL.
7. Backend exchanges the code for tokens and stores the merchant connection.
8. Future sync runs use the stored refresh token automatically.

No manual token generation or manual link-sending is required for normal onboarding.

## Token Lifecycle

### Initial connection

- Square returns `access_token`, `refresh_token`, `merchant_id`, `scope`, and expiry metadata.
- Tokens are stored in `square_connections` in the auth SQLite database.
- Tokens are encrypted using `TOKEN_ENCRYPTION_KEY`.

### Expired access tokens

When Square rejects an API call with an auth error:

1. Backend reads the stored refresh token.
2. Backend calls Square `POST /oauth2/token` with `grant_type=refresh_token`.
3. Square returns a new access token and may also return a replacement refresh token.
4. Backend updates the stored connection.
5. Backend retries the sync call.

User action is only required if the Square connection is revoked or the refresh token is no longer valid.

## Runtime Storage

### Auth database

Default path:

- `artifacts/runtime/auth.db`

Tables:

- `users`
- `user_sessions`
- `oauth_states`
- `square_connections`

### Existing sync database

Default path:

- `artifacts/data/eod_sync.db`

This still stores sync run history and alerts. It is not the source of truth for OAuth tokens.

## Environment Variables

### Required for auth

- `APP_AUTH_SECRET`
  - HMAC secret used to hash session tokens before DB storage.
- `TOKEN_ENCRYPTION_KEY`
  - Fernet key used to encrypt Square tokens at rest.

Generate a Fernet key with Python:

```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

### Required for Square OAuth

- `SQUARE_CLIENT_ID`
- `SQUARE_CLIENT_SECRET`
- `SQUARE_REDIRECT_URI`
- `SQUARE_ENV`
  - `production` or `sandbox`

### Optional

- `SQUARE_SCOPES`
  - Default: `MERCHANT_PROFILE_READ ORDERS_READ ITEMS_READ PAYMENTS_READ`
- `MILK_AUTH_DB_PATH`
  - Default: `artifacts/runtime/auth.db`
- `MILK_SESSION_COOKIE_SECURE`
  - Set to `true` in HTTPS production deployments.

### Existing sync/runtime variables still used

- `MILK_SALES_PATH`
- `MILK_INGREDIENTS_PATH`
- `MILK_SYNC_DB_PATH`
- `MILK_LOCAL_TIMEZONE`
- `MILK_EOD_DEFAULT_LOOKBACK_DAYS`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_MODEL_NAME`
- `MLFLOW_MODEL_ALIAS`

## API Surface

### App auth

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`

Request body for register/login:

```json
{
  "email": "owner@example.com",
  "password": "strong-password"
}
```

### Square auth

- `GET /api/auth/square/connect`
- `GET /api/auth/square/callback`
- `GET /api/auth/square/status`
- `POST /api/auth/square/disconnect`

### User-scoped sync

- `GET /api/integrations/square/eod/latest`
- `POST /api/integrations/square/eod-sync`

These endpoints require a valid app session cookie.

## Square Developer Dashboard Setup

### One-time setup

1. Create or use your Square application in the Square Developer Dashboard.
2. Set the correct production redirect URI to:
   - `https://<your-domain>/api/auth/square/callback`
3. Ensure the same redirect URI is present in `SQUARE_REDIRECT_URI`.
4. Confirm scopes match the data you need.

### If you change domains

You must update the redirect URI in both:

- Square Developer Dashboard
- deployment environment variable `SQUARE_REDIRECT_URI`

## Deployment Checklist

1. Install dependencies from `pyproject.toml`.
2. Set `APP_AUTH_SECRET`.
3. Set `TOKEN_ENCRYPTION_KEY`.
4. Set `SQUARE_CLIENT_ID`.
5. Set `SQUARE_CLIENT_SECRET`.
6. Set `SQUARE_REDIRECT_URI`.
7. Set `SQUARE_ENV=production`.
8. Set `MILK_SESSION_COOKIE_SECURE=true` when running behind HTTPS.
9. Ensure the app can write to:
   - `artifacts/runtime/`
   - `artifacts/data/`
10. Start the API:

```bash
uvicorn milk_dashboard.api.app:app --host 0.0.0.0 --port 8000
```

## Operational Notes

### What the developer still does

- Maintain Square app credentials.
- Maintain valid redirect URIs for each deployed domain.
- Support clients only when they revoke access or misconfigure their own Square permissions.

### What the developer no longer does

- Manually create auth links for each client.
- Manually collect/store tokens.
- Manually refresh expired access tokens.

## Failure Cases

### User must reconnect if

- They revoked your app in Square.
- Square no longer accepts the stored refresh token.
- Your Square app credentials or redirect URI no longer match the deployment.

### Sync fails if

- Auth env vars are missing.
- Token encryption key is missing or invalid.
- Sales or ingredient file paths are invalid.
- Square API permissions are insufficient for the requested data.

## Security Notes

- Session cookies are HTTP-only.
- Session tokens are hashed before DB storage.
- Square tokens are encrypted at rest.
- Access tokens are not returned from API responses.
- Production should always use HTTPS with secure cookies enabled.
