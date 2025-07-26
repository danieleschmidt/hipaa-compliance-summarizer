# Rerere Cache Audit

This directory contains Git rerere cache entries for audit purposes.

## What is Rerere?

Git's `rerere` (reuse recorded resolution) functionality records how merge conflicts are resolved and automatically applies the same resolution when the same conflict occurs again.

## Audit Process

Run the following command to view current rerere cache:

```bash
git rerere diff
```

## Cache Entries

Cache entries will be committed here when conflicts are resolved to provide visibility into automatic conflict resolution patterns.

## Files

- `conflict-resolutions.log` - Log of resolved conflicts
- Individual cache files as they're created by Git rerere

## Security Note

All rerere resolutions should be reviewed to ensure they don't introduce security issues through automatic conflict resolution.