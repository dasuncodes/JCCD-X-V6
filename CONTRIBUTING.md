# Contributing to JCCD-X-V6

Thank you for your interest in contributing to JCCD-X-V6!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/JCCD-X-V6.git`
3. Create a new branch for your feature or fix

## Branch Naming Structure

Use the following format for branch names:

```
<type>/<ticket-id>-<short-description>
```

### Types

| Type | Description |
|------|-------------|
| `feature/` | New features |
| `fix/` | Bug fixes |
| `hotfix/` | Urgent production fixes |
| `refactor/` | Code refactoring |
| `docs/` | Documentation changes |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

### Examples

```
feature/123-user-authentication
fix/456-login-redirect-issue
hotfix/789-security-patch
refactor/101-cleanup-database-queries
docs/202-update-api-documentation
```

## Commit Message Structure

Follow this format for commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

Same as branch types: `feat`, `fix`, `hotfix`, `refactor`, `docs`, `test`, `chore`

### Examples

```
feat(auth): add OAuth2 login support

- Implement Google OAuth2 provider
- Add user session management
- Update login page UI

Closes #123
```

```
fix(api): resolve null pointer exception in user endpoint

- Add null check before accessing user object
- Return appropriate error response

Fixes #456
```

### Rules

- Use imperative mood: "add feature" not "added feature"
- Subject line: max 50 characters
- Body: wrap at 72 characters
- Reference issues in footer using `Closes #`, `Fixes #`, or `Resolves #`

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Keep PRs focused and reasonably sized
4. Fill out the PR template with all relevant details
5. Request review from maintainers

## Code Style

- Follow existing code conventions in the project
- Write meaningful commit messages
- Add comments only when necessary
- Keep functions focused and modular
