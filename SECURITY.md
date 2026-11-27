# Security Policy

## Supported versions

sparkfit is a small, dependency-free command-line tool. Security fixes are applied
to the latest released version.

| Version | Supported |
| ------- | --------- |
| 0.6.x   | yes       |
| < 0.6   | no        |

## Reporting a vulnerability

Please do not open a public issue for security reports. Instead, use GitHub's
private vulnerability reporting on this repository
(Security tab -> Report a vulnerability), or email the maintainer at
francesco.delre@protonmail.com.

Please include enough detail to reproduce the problem. You can expect an initial
response within a few days. Once a fix is available, we will publish it and credit
the reporter unless anonymity is requested.

## Scope notes

sparkfit performs local computation only. The single network feature is the
optional Hugging Face auto-fetch, which issues a read-only HTTPS request to
`huggingface.co` to download a model `config.json`. If `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`) is set, it is sent to `huggingface.co` only, as a bearer
token, so gated models can be read. No telemetry is collected and no data leaves
the machine otherwise.

The only file sparkfit writes is its own config (`~/.sparkfitrc`, or
`$SPARKFIT_CONFIG`), and only when you run `sparkfit doctor --save`. On the device
it runs `nvidia-smi` read-only queries for live memory and temperature.
