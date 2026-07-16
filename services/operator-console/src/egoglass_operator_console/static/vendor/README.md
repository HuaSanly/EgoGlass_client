# Vendored browser dependencies

These unmodified browser builds are committed so the Windows desktop client
does not require internet access at runtime.

| File | Upstream | Version | License | SHA-256 |
| --- | --- | --- | --- | --- |
| `three.module-0.185.1.min.js` | https://github.com/mrdoob/three.js | 0.185.1 | MIT | `86BCEE248B64F44BCFC23C331AE74619061957D59CAB040171DCB6FB5900BEB6` |
| `three.core.min.js` | https://github.com/mrdoob/three.js | 0.185.1 | MIT | `05B2609338C76CD65DAF74F3AC515BC9A5045E1B3B33EDC07D8C9BD55250FA90` |
| `ahrs-1.3.3.js` | https://github.com/psiphi75/ahrs | 1.3.3 | Apache-2.0 | `9EE207F4C8CB3A5A29AAECDDB16E51A515CB99D70728FCBB63D1439791FC0FAE` |

The exact license texts are stored beside the browser builds. The AHRS package
manifest contains a stale `APSL-2.0` identifier, while its bundled `LICENSE`,
source headers, and upstream repository identify Apache-2.0. EgoGlass relies on
the included source license.
