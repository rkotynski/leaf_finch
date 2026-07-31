# Security policy

LEAF_FINCH processes local configuration files, numerical arrays, and PyTorch checkpoints. Do not load checkpoints from untrusted sources, particularly with PyTorch versions that do not support restricted `weights_only` loading.

Please report security-sensitive issues privately to the repository maintainers rather than opening a public issue. Include the affected version, operating system, Python and PyTorch versions, reproduction steps, and the potential impact.
